# Check if variant sent_ids match fold sent_ids
python3 -c "
import pickle
# Load variant sent_ids
variants = pickle.load(open('data/processed/all_variants_final.pkl','rb'))
var_ids = set(v['sent_id'] for v in variants)

# Load fold sent_ids  
fold_ids = set()
for i in range(5):
    ids = pickle.load(open(f'data/processed/pcfg_folds/fold_{i}/test_sent_ids.pkl','rb'))
    fold_ids.update(ids)

matched = var_ids & fold_ids
print(f'Variant sent_ids: {len(var_ids)}')
print(f'Fold sent_ids: {len(fold_ids)}')
print(f'Matched: {len(matched)}')
print(f'Unmatched variant ids (first 5): {list(var_ids - fold_ids)[:5]}')
print(f'Sample variant id: {list(var_ids)[0]}')
print(f'Sample fold id: {list(fold_ids)[0]}')
"