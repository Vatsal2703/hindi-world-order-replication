"""Repo-root resolution shared by the scripts/ pipeline."""

import os

REPO_NAME = "hindi-world-order-replication"


def find_base(start_path):
    """Walk up from `start_path` (pass __file__) to find the repo root."""
    d = os.path.dirname(os.path.abspath(start_path))
    while d and d != os.path.dirname(d):
        if os.path.basename(d) == REPO_NAME:
            return d
        if os.path.isdir(os.path.join(d, REPO_NAME)):
            return os.path.join(d, REPO_NAME)
        d = os.path.dirname(d)
    return "."
