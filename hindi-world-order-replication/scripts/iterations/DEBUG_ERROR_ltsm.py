import pickle
with open("./data/processed/all_variants_final.pkl", 'rb') as f:
    data = pickle.load(f)

# Look at the first group (train-s6 has 5 variants)
group_s6 = [item for item in data if item['sent_id'] == 'train-s6']
for i, item in enumerate(group_s6):
    print(f"Index {i} | Reference: {item['is_reference']} | Order: {item['variant_order'][:5]}...")