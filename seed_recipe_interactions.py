import os, json, random, pandas as pd

BASE     = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, 'data')
INT_F    = os.path.join(DATA_DIR, 'recipe_interactions.csv')
REC_F    = os.path.join(DATA_DIR, 'Recipes_Featured.csv')

users = [f"user_{i}" for i in range(1,16)]


recipes = pd.read_csv(REC_F).rename(columns=lambda c: c.strip())
if 'Recipe Id' in recipes.columns:
    recipes.rename(columns={'Recipe Id':'RecipeID'}, inplace=True)
rids = recipes['RecipeID'].dropna().astype(int).tolist()

rows = []
random.seed(42)
for u in users:

    if len(rids) >= 30:
        picks = random.sample(rids, 30)
    else:
        picks = [random.choice(rids) for _ in range(30)]
    for rid in picks:
        rows.append({
            'user_id': u,
            'recipe_id': rid,
            'interaction': random.randint(1,5)
        })

df = pd.DataFrame(rows, columns=['user_id','recipe_id','interaction'])
df.to_csv(INT_F, index=False)
print(f"Seeded {len(users)} users × 30 interactions = {len(df)} rows into {INT_F}")
