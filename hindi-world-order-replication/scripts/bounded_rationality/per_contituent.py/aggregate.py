"""
Per-constituent surprisal aggregation.

Takes per-token surprisal from any estimator and buckets it into preverbal
constituents, keyed by position. This is the missing piece between the
sentence-total surprisal MTP-1 produced and the positional predictors the
2023 bounded-rationality paper needs (last-constituent's measure, 2nd-last,
etc. — Tables 1 and 3 of Ranjan & von der Malsburg 2023).

The aggregator is estimator-agnostic: it takes a list[float] of per-token
surprisal and a Sentence + variant_order. Works for the trigram now; will
work unchanged for LSTM/LLM later.

Aggregation choice: we emit BOTH sum and mean within each constituent.
- sum scales with constituent length (cumulative cognitive load)
- mean controls for length (per-word predictability)
The paper doesn't specify which to use for surprisal; emitting both lets
the downstream classifier choose without a re-run.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence


# ============================================================================
# CONSTITUENT-MEMBERSHIP DERIVATION
# ============================================================================
# We don't store subtree blocks in the variant output (script 04 throws
# them away after permuting), so we re-derive them here from the original
# Sentence. This keeps the aggregator self-contained and avoids touching
# script 04.

def _subtree_indices(head_idx: int, sentence) -> set[int]:
    """Iterative DFS — same logic as script 04's get_subtree_indices."""
    visited: set[int] = set()
    stack = [head_idx]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for w in sentence.words:
            if w.head == cur:
                stack.append(w.idx)
    return visited


def preverbal_blocks(sentence) -> list[tuple[int, set[int]]]:
    """
    Return [(head_idx, subtree_token_indices), ...] for each preverbal
    direct dependent of the root verb, in original left-to-right order.

    Matches the preverbal-dep definition in script 04 (and the 2022 paper):
      head == root_idx, idx < root_idx, deprel != 'punct'.

    If the project ends up using the broader 2023 definition, this is the
    one place to change it.
    """
    root = sentence.root_word
    if root is None:
        return []
    root_idx = root.idx
    preverbal_heads = sorted(
        [w for w in sentence.words
         if w.head == root_idx
         and w.idx < root_idx
         and w.deprel != "punct"],
        key=lambda w: w.idx,
    )
    return [(h.idx, _subtree_indices(h.idx, sentence)) for h in preverbal_heads]


# ============================================================================
# AGGREGATION
# ============================================================================

@dataclass
class ConstituentScores:
    """One preverbal constituent's measurements, in a variant's order."""
    position_in_variant: int        # 1 = leftmost preverbal in this variant, ..., k = verb-adjacent
    position_from_verb: int         # -k .. -1; -1 = verb-adjacent. Signed for later postverbal work.
    head_idx: int                   # token idx of constituent head in original sentence
    token_indices: list[int] = field(default_factory=list)
    length: int = 0                 # word count = constituent length (paper's const_length)
    surprisal_sum: float = 0.0      # sum of per-word surprisal within the constituent
    surprisal_mean: float = 0.0     # mean (controls for length)


@dataclass
class SentenceAggregation:
    """Aggregation output for one (sentence, variant) pair."""
    sent_id: str
    is_reference: bool
    k: int                                  # number of preverbal constituents
    sentence_total_surprisal: float         # = sum(per_token_surprisal); backward-compat with MTP-1
    constituents: list[ConstituentScores]   # ordered as they appear in the variant

    # convenience accessors keyed the way the paper writes its tables
    def by_position_from_verb(self, p: int) -> ConstituentScores | None:
        for c in self.constituents:
            if c.position_from_verb == p:
                return c
        return None

    def last(self) -> ConstituentScores | None:
        return self.by_position_from_verb(-1)

    def second_last(self) -> ConstituentScores | None:
        return self.by_position_from_verb(-2)

    def to_flat_row(self) -> dict:
        """Flat dict for DataFrame export — one row per (sent, variant)."""
        row = {
            "sent_id": self.sent_id,
            "is_reference": self.is_reference,
            "k": self.k,
            "sentence_total_surprisal": self.sentence_total_surprisal,
        }
        # paper-style numbered columns: const1 = leftmost preverbal, const_k = verb-adjacent
        for c in self.constituents:
            i = c.position_in_variant
            row[f"const{i}_surp_sum"]  = c.surprisal_sum
            row[f"const{i}_surp_mean"] = c.surprisal_mean
            row[f"const{i}_length"]    = c.length
        # named convenience slots (paper's main predictors)
        last = self.last()
        s2l  = self.second_last()
        if last:
            row["last_surp_sum"]  = last.surprisal_sum
            row["last_surp_mean"] = last.surprisal_mean
            row["last_length"]    = last.length
        if s2l:
            row["second_last_surp_sum"]  = s2l.surprisal_sum
            row["second_last_surp_mean"] = s2l.surprisal_mean
            row["second_last_length"]    = s2l.length
        return row


def aggregate(
    sentence,
    variant_order: Sequence[int],
    per_token_surprisal: Sequence[float],
    *,
    sent_id: str | None = None,
    is_reference: bool = False,
) -> SentenceAggregation:
    """
    Bucket per-token surprisal into preverbal constituents.

    Alignment contract (important):
      per_token_surprisal[i] is the surprisal of the token at
      variant_order[i] in the original sentence. The estimator must have
      been called on tokens drawn from variant_order in the same order.

    Tokens NOT in any preverbal block (the verb itself + postverbal + non-
    preverbal-dependent material) are silently ignored for constituent
    bucketing but still counted in sentence_total_surprisal.
    """
    if len(variant_order) != len(per_token_surprisal):
        raise ValueError(
            f"variant_order ({len(variant_order)}) and per_token_surprisal "
            f"({len(per_token_surprisal)}) must align 1:1. The estimator "
            f"must score tokens in variant_order."
        )

    # Build token_idx -> block_id (block_id = head_idx of the preverbal constituent)
    blocks = preverbal_blocks(sentence)
    token_to_head: dict[int, int] = {}
    for head_idx, tok_indices in blocks:
        for t in tok_indices:
            token_to_head[t] = head_idx
    k = len(blocks)

    # Walk variant_order; group consecutive runs of the same block_id.
    # Detecting "first time we see this block" gives the block's left-to-right
    # position IN THE VARIANT (1-indexed).
    grouped: dict[int, list[int]] = {}    # head_idx -> per-token-surprisal indices in this run
    first_seen_order: list[int] = []      # head_idx in the order they first appear in variant_order
    for surp_i, tok_idx in enumerate(variant_order):
        head = token_to_head.get(tok_idx)
        if head is None:
            continue  # verb / postverbal / non-preverbal-dependent
        if head not in grouped:
            grouped[head] = []
            first_seen_order.append(head)
        grouped[head].append(surp_i)

    # Assemble ConstituentScores in variant order
    constituents: list[ConstituentScores] = []
    for variant_pos, head_idx in enumerate(first_seen_order, start=1):
        surp_indices = grouped[head_idx]
        tok_indices  = [variant_order[i] for i in surp_indices]
        surps        = [per_token_surprisal[i] for i in surp_indices]
        length       = len(tok_indices)
        s_sum        = float(sum(surps))
        s_mean       = s_sum / length if length > 0 else 0.0
        # position_from_verb: verb-adjacent (variant_pos == k) is -1
        pos_from_verb = -(k - variant_pos + 1)
        constituents.append(ConstituentScores(
            position_in_variant=variant_pos,
            position_from_verb=pos_from_verb,
            head_idx=head_idx,
            token_indices=tok_indices,
            length=length,
            surprisal_sum=s_sum,
            surprisal_mean=s_mean,
        ))

    return SentenceAggregation(
        sent_id=sent_id or getattr(sentence, "sent_id", ""),
        is_reference=is_reference,
        k=k,
        sentence_total_surprisal=float(sum(per_token_surprisal)),
        constituents=constituents,
    )