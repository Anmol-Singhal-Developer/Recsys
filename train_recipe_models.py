
import os
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
from lightfm import LightFM
from lightfm.data import Dataset as LFMDataset
import xgboost as xgb
from joblib import dump
from lightfm.evaluation import precision_at_k, auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_DIR    = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODEL_DIR   = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

REC_CSV     = os.path.join(DATA_DIR, 'Recipes_Featured.csv')
INT_CSV     = os.path.join(DATA_DIR, 'recipe_interactions.csv')

CF_OUT      = os.path.join(MODEL_DIR, 'recipe_cf.joblib')
LF_OUT      = os.path.join(MODEL_DIR, 'recipe_lightfm.joblib')
XGB_OUT     = os.path.join(MODEL_DIR, 'recipe_xgb_ranker.model')

def load_data():
    print("[1/4] Loading data…")
    recipes = pd.read_csv(REC_CSV).rename(columns=lambda c: c.strip())
    if 'Recipe Id' in recipes.columns:
        recipes.rename(columns={'Recipe Id':'RecipeID'}, inplace=True)
    interactions = pd.read_csv(INT_CSV)
    return recipes, interactions

def train_and_eval_cf(inter_df):
    print(f"[2/4][CF] {len(inter_df)} interaction rows found.")
    if len(inter_df) < 2:
        print("[  CF] Too few interactions — skipping CF.")
        return None

    reader   = Reader(rating_scale=(1,5))
    data     = Dataset.load_from_df(inter_df[['user_id','recipe_id','interaction']], reader)

    trainset, testset = surprise_train_test_split(data, test_size=0.1, random_state=42)
    algo = SVD(n_factors=100, reg_all=0.05, lr_all=0.005, n_epochs=30)
    print("[  CF] Fitting SVD…")
    algo.fit(trainset)

    preds = algo.test(testset)
    rmse  = accuracy.rmse(preds, verbose=False)
    mae   = accuracy.mae(preds, verbose=False)
    print(f"[  CF] RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    dump(algo, CF_OUT)
    print(f"[  CF] Model saved to {CF_OUT}")
    return algo

def train_lightfm(recipes_df, inter_df):
    users = inter_df['user_id'].astype(str).unique().tolist()
    items = inter_df['recipe_id'].astype(str).unique().tolist()
    print(f"[3/4][LF] {len(users)} users, {len(items)} interacted items.")

    if len(users) < 3 or len(items) < 5:
        print(f"[  LF] Only {len(users)} users or {len(items)} items — skipping LightFM.")
        return None

    ds = LFMDataset()
    ds.fit(users=users, items=items)

    interactions, _ = ds.build_interactions([
        (str(u), str(i), int(r))
        for u, i, r in zip(
            inter_df['user_id'],
            inter_df['recipe_id'],
            inter_df['interaction']
        )
    ])
    print(f"[  LF] Built interactions matrix of shape {interactions.shape}")

    subset = recipes_df[recipes_df['RecipeID'].astype(str).isin(items)]
    item_feats = {}
    for _, row in subset.iterrows():
        rid   = str(row['RecipeID'])
        feats = [
            c.strip() for c in str(row.get('Cuisine','')).split(',')
            if c.strip()
        ] + [
            d.strip() for d in str(row.get('Dietary','')).split(',')
            if d.strip()
        ]
        if not feats:
            feats = ['Unknown']
        item_feats[rid] = feats

    all_feats = sorted({f for feats in item_feats.values() for f in feats})
    print(f"[  LF] Prepared {len(item_feats)} items with {len(all_feats)} unique features")

    ds.fit(users=users, items=list(item_feats.keys()), item_features=all_feats)

    item_feat_mat = ds.build_item_features(item_feats.items())
    print(f"[  LF] Built item-feature matrix of shape {item_feat_mat.shape}")

    lf_model = LightFM(loss='logistic', no_components=50, learning_rate=0.05)
    print("[  LF] Training LightFM (logistic loss)…")
    lf_model.fit(interactions, item_features=item_feat_mat,
                 epochs=10, num_threads=2)
    dump(lf_model, LF_OUT)
    print(f"[  LF] Model saved to {LF_OUT}")
    
    prec5 = precision_at_k(lf_model, interactions,
                          item_features=item_feat_mat, k=5).mean()
    auc   = auc_score(lf_model, interactions,
                      item_features=item_feat_mat).mean()
    print(f"[  LF] Precision@5: {prec5:.4f}, AUC: {auc:.4f}")

    return lf_model


def train_xgb_ranker(recipes_df, inter_df):
    print(f"[4/4][XGB] Preparing data…")
    if len(inter_df) < 10:
        print("[ XGB] Too few interactions — skipping XGBoost.")
        return None

    merged = inter_df.merge(
        recipes_df[['RecipeID','Cuisine','Dietary']].rename(columns={'RecipeID':'recipe_id'}),
        on='recipe_id', how='left'
    )
    merged['n_cuisine'] = merged['Cuisine'].fillna('').apply(lambda s: len(s.split(',')))
    merged['n_dietary'] = merged['Dietary'].fillna('').apply(lambda s: len(s.split(',')))

    feat_cols = ['interaction','n_cuisine','n_dietary']
    X = merged[feat_cols].values
    y = merged['interaction'].astype(int).values

    dtrain = xgb.DMatrix(X, label=y, feature_names=feat_cols)
    params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg',
    'eta': 0.05,
    'max_depth': 4,
    'lambda': 1.0,
    'min_child_weight': 10
    }
    print("[  XGB] Training ranker…")
    bst = xgb.train(params, dtrain, num_boost_round=100)
    bst.save_model(XGB_OUT)
    print(f"[  XGB] Model saved to {XGB_OUT}")

    preds = bst.predict(dtrain)
    
    mae  = mean_absolute_error(y, preds)
    print(f"[  XGB]  MAE: {mae:.4f}")

    return bst

if __name__ == "__main__":
    recipes_df, inter_df = load_data()

    print("\n[INFO] Starting recipe‐model training…")
    cf = train_and_eval_cf(inter_df)
    lf = train_lightfm(recipes_df, inter_df)
    xbm = train_xgb_ranker(recipes_df, inter_df)

    print("\n[INFO] Final Summary:")
    print("CF model trained:    ", cf is not None)
    print("LightFM model trained:", lf is not None)
    print("XGB ranker trained:   ", xbm is not None)
