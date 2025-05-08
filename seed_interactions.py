import pandas as pd
import numpy as np
import random


restaurants_df = pd.read_csv('data/Restaurants_Featured.csv')  
restaurant_ids = restaurants_df['ResId'].dropna().astype(str).unique().tolist()


num_users = 300
likes_per_user = 10
dislikes_per_user = 5


interactions = []
for i in range(num_users):
    user_id = f"user_{i+1}"
    liked_ids = set(random.sample(restaurant_ids, likes_per_user))
    remaining = list(set(restaurant_ids) - liked_ids)
    disliked_ids = set(random.sample(remaining, dislikes_per_user))

    for rid in liked_ids:
        interactions.append((user_id, rid, 1))
    for rid in disliked_ids:
        interactions.append((user_id, rid, 0))

interactions_df = pd.DataFrame(interactions, columns=["user_id", "ResId", "interaction"])
interactions_df.to_csv('data/interactions.csv', index=False)

print(f"Synthetic interactions generated for {num_users} users.")
