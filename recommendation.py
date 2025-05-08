
import os, ast, json, datetime, pandas as pd, numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
from lightfm import LightFM
from lightfm.data import Dataset as LFMDataset
import xgboost as xgb
from joblib import dump, load

BASE_DIR       = os.path.dirname(__file__)
DATA_DIR       = os.path.join(BASE_DIR, 'data')
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


INTERACTIONS_F    = os.path.join(DATA_DIR, 'interactions.csv')
USER_PROFILES_F   = os.path.join(DATA_DIR, 'user_profiles.json')
CF_MODEL_F        = os.path.join(MODELS_DIR, 'cf_svd.joblib')
SCALER_F          = os.path.join(MODELS_DIR, 'cb_scaler.joblib')
LF_MODEL_F        = os.path.join(MODELS_DIR, 'lightfm_model.joblib')
XGB_MODEL_F       = os.path.join(MODELS_DIR, 'xgb_ranker.model')
XGB_FEATURES_F    = os.path.join(MODELS_DIR, 'xgb_features.json')


def parse_list_column(x):
    if isinstance(x, list): return x
    if isinstance(x, str) and x.startswith('['):
        try: return ast.literal_eval(x)
        except: return []
    return []


def load_restaurant_df():
    df = pd.read_csv(os.path.join(DATA_DIR, 'Restaurants_Featured.csv'))
    df['CuisineList'] = df['CuisineList'].apply(parse_list_column)
    return df


def load_user_profiles():
    if os.path.exists(USER_PROFILES_F):
        with open(USER_PROFILES_F) as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_user_profiles(profiles):
    with open(USER_PROFILES_F, 'w') as f:
        json.dump(profiles, f, indent=2)

def get_user_profile(user_id):
    return load_user_profiles().get(user_id, {})

def update_user_profile(user_id, **kwargs):
    profiles = load_user_profiles()
    user = profiles.get(user_id, {})
    user.update(kwargs)
    profiles[user_id] = user
    save_user_profiles(profiles)



def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * (2 * asin(sqrt(a)))



def record_interactions(user_id, liked_resid_list):
    if not liked_resid_list: return
    rows = [(user_id, rid, 1) for rid in liked_resid_list]
    df0 = pd.DataFrame(rows, columns=['user_id','ResId','interaction'])
    if os.path.exists(INTERACTIONS_F):
        df_hist = pd.read_csv(INTERACTIONS_F, dtype=str)
        df_all  = pd.concat([df_hist, df0], ignore_index=True)
        df_all.drop_duplicates(['user_id','ResId'], inplace=True)
    else:
        df_all = df0
    df_all.to_csv(INTERACTIONS_F, index=False)

def train_and_eval_cf(test_size=0.2):
    try:
        df_int = pd.read_csv(INTERACTIONS_F)
        n = len(df_int)
        if n < 2:
            print(f"[CF] Only {n} interaction(s) found – skipping CF training.")
            return None

        df_int = df_int.astype(str)
        reader = Reader(rating_scale=(0,1))
        data   = Dataset.load_from_df(df_int[['user_id','ResId','interaction']], reader)

        test_n = max(int(test_size * n), 1)
        if test_n >= n:
            test_n = n - 1

        trainset, testset = surprise_train_test_split(
            data, test_size=test_n, random_state=42
        )
        algo = SVD(n_factors=50, reg_all=0.02, lr_all=0.005, random_state=42)
        algo.fit(trainset)
        dump(algo, CF_MODEL_F)

        preds = algo.test(testset)
        accuracy.rmse(preds, verbose=True)
        accuracy.mae(preds, verbose=True)

        print("[CF] Training complete successfully.")
        y_true = [int(p.r_ui) for p in preds]
        y_score = [float(p.est) for p in preds]
        print("[CF] Evaluation:", evaluate_model(y_true, y_score, k=10))
        return algo

    except Exception as e:
        print("[CF ERROR]", e)
        return None



def score_cf_surprise(algo, df, user_id):
    if algo is None or not os.path.exists(CF_MODEL_F):
        return np.zeros(len(df))
    if isinstance(algo, str):
        algo = load(CF_MODEL_F)
    return np.array([ algo.predict(user_id, str(r)).est for r in df['ResId'] ])



def prepare_content_features(df, price_bin, cost_bin, hours_open):
    mlb = MultiLabelBinarizer()
    cuisines_ohe = pd.DataFrame(
        mlb.fit_transform(df['CuisineList']),
        columns=[f'cuisine_{c}' for c in mlb.classes_], index=df.index
    )

    df['rating_score'] = df['AggregateRating'].fillna(0)/5.0
    df['votes_score']  = np.log1p(df['Votes'].fillna(0))

    est_ohe = pd.get_dummies(df['Establishment'].fillna('Unknown'), prefix='est')
    loc_ohe = pd.get_dummies(df['Locality'].fillna('Unknown'),   prefix='loc')

    df['log_cost2'] = np.log1p(df['AverageCostForTwo'].fillna(0))
    df['cost_pp']   = df['AverageCostForTwo'].fillna(0)/2.0

    now_h = datetime.datetime.now().hour
    df['is_open_now'] = (
        (df['OpeningHour'].fillna(-1)<=now_h) &
        (now_h<df['ClosingHour'].fillna(24))
    ).astype(int)

    hcols = [c for c in df.columns if c.startswith('Highlight_')]
    df['highlight_count'] = df[hcols].sum(axis=1).astype(int)

    other_bins = ['OpenDays','IsOpenAllWeek','IsWeekendOpen']
    other_nums = ['CityPopularity']
    others     = df[other_bins+other_nums].fillna(0)

    df['price_bin'], df['cost_bin'], df['hours_open'] = price_bin, cost_bin, hours_open

    scaler = MinMaxScaler()
    X_num  = scaler.fit_transform(df[['price_bin','cost_bin','hours_open']])
    dump(scaler, SCALER_F)

    X = np.hstack([
        cuisines_ohe.values,
        df[['rating_score','votes_score']].values,
        est_ohe.values, loc_ohe.values,
        df[['log_cost2','cost_pp']].values,
        df[['is_open_now','highlight_count']].values,
        df[hcols].fillna(0).values,
        others.values,
        X_num
    ])

    feature_names = (
        list(cuisines_ohe.columns) +
        ['rating_score','votes_score'] +
        list(est_ohe.columns) +
        list(loc_ohe.columns) +
        ['log_cost2','cost_pp','is_open_now','highlight_count'] +
        hcols + other_bins + other_nums + ['price_bin','cost_bin','hours_open']
    )

    df2 = df.drop(columns=[
        'CuisineList','AggregateRating','Votes','Establishment','Locality',
        'AverageCostForTwo','OpeningHour','ClosingHour'
    ] + hcols + other_bins + other_nums + ['price_bin','cost_bin','hours_open'],
             errors='ignore')

    return df2, X, scaler, mlb.classes_.tolist(), feature_names


def score_content(df, X, scaler, feature_names,
                  cuisines, wants, price_bin, cost_bin, hours_open):
    u = np.zeros(len(feature_names))
    for c in cuisines:
        key = f'cuisine_{c}'
        if key in feature_names:
            u[feature_names.index(key)] = 1
    for a in wants:
        key = f'Highlight_{a}'
        if key in feature_names:
            u[feature_names.index(key)] = 1

    vals = pd.DataFrame([[price_bin, cost_bin, hours_open]],
                    columns=['price_bin', 'cost_bin', 'hours_open'])
    scaled = scaler.transform(vals)[0]

    offset = len(feature_names) - 3
    u[offset:offset+3] = scaled

    return cosine_similarity([u], X)[0]




def train_lightfm(no_components=30, epochs=5):
    from joblib import dump
    try:
        df_int = pd.read_csv(INTERACTIONS_F).astype(str)
        if df_int.empty:
            print("[LightFM] No interactions – skipping LightFM training.")
            return None

        user_counts = df_int['user_id'].nunique()
        item_counts = df_int['ResId'].nunique()

        if user_counts < 3 or item_counts < 5:
            print("[LightFM] Not enough users or items for training. Skipping.")
            return None

        lfm_ds = LFMDataset()
        users = df_int['user_id'].unique()
        items = df_int['ResId'].unique()
        lfm_ds.fit(users, items)

        u_i, _ = lfm_ds.build_interactions(
            [(r.user_id, r.ResId, int(r.interaction)) for r in df_int.itertuples()]
        )

        model = LightFM(no_components=no_components, loss='logistic')
        model.fit(u_i, epochs=epochs, num_threads=2)

        dump(model, LF_MODEL_F)
        print("[LightFM] Model trained and saved successfully.")
        
        y_true = []
        y_score = []
        for user_id in df_int['user_id'].unique():
            pos_items = df_int[df_int['user_id'] == user_id]['ResId'].tolist()
            all_items = df_int['ResId'].unique().tolist()

            user_idx = lfm_ds.mapping()[0].get(user_id)
            item_map = lfm_ds.mapping()[2]

            if user_idx is None:
                continue

            for rid in all_items:
                item_idx = item_map.get(rid)
                if item_idx is not None:
                    y_score.append(model.predict([user_idx], [item_idx])[0])
                    y_true.append(1 if rid in pos_items else 0)

        print("[LightFM] Evaluation:", evaluate_model(y_true, y_score, k=10))

        return model

    except Exception as e:
        print("[LightFM ERROR]", str(e))
        return None





def score_lightfm(model, df, user_id):

     if model is None or not os.path.exists(LF_MODEL_F):
         return np.zeros(len(df))

     df_int = pd.read_csv(INTERACTIONS_F).astype(str)
     ds     = LFMDataset()
     ds.fit(df_int['user_id'].unique(), df_int['ResId'].unique())

     maps     = ds.mapping()
     user_map = maps[0]; item_map = maps[2]

     uidx   = user_map.get(user_id)
     scores = []
     for rid in df['ResId'].astype(str):
         iidx = item_map.get(rid)
         if uidx is None or iidx is None:
             scores.append(0.0)
         else:

           try:
               scores.append(model.predict([uidx], [iidx])[0])
           except ValueError:
               scores.append(0.0)
     return np.array(scores)




def train_xgb_ranker():
    try:
        df_int = pd.read_csv(INTERACTIONS_F).astype({'user_id': str, 'ResId': str})
        if df_int.empty:
            print("[XGB] No interactions – skipping XGBoost training.")
            return None

        df_rest = load_restaurant_df()
        df_feat, _, _, _, feat_names = prepare_content_features(
            df_rest, price_bin=0, cost_bin=0, hours_open=0
        )

        data = df_int[['user_id', 'ResId', 'interaction']].rename(columns={'interaction': 'label'})
        merged = data.merge(
            df_feat.assign(ResId=df_rest['ResId'].astype(str)),
            on='ResId', how='left'
        )

        common_feats = [f for f in feat_names if f in merged.columns]
        if not common_feats:
            print("[XGB] No matching features in merged data – skipping.")
            return None

        y = merged['label'].values
        X = merged[common_feats].values

        dtrain = xgb.DMatrix(X, label=y, feature_names=common_feats)
        params = {
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg',
            'eta': 0.1,
            'max_depth': 6,
            'min_child_weight': 10
        }
        model = xgb.train(params, dtrain, num_boost_round=100)

        model.save_model(XGB_MODEL_F)
        with open(XGB_FEATURES_F, 'w') as f:
            json.dump(common_feats, f)
        
        y_pred = model.predict(dtrain)
        print("[XGB] Evaluation:", evaluate_model(y, y_pred, k=10))

        print(f"[XGB] Trained on {len(common_feats)} features; saved model.")
        return model

    except Exception as e:
        print("[XGB ERROR]", str(e))
        return None



def score_xgb(model, df):

    if model is None or not os.path.exists(XGB_MODEL_F) or not os.path.exists(XGB_FEATURES_F):
        return np.zeros(len(df))

    try:
        with open(XGB_FEATURES_F, 'r') as f:
            feat_names = json.load(f)
    except:
        return np.zeros(len(df))

    try:
        df_feat, X, _, _, _ = prepare_content_features(
            df, price_bin=0, cost_bin=0, hours_open=0
        )
        dtest = xgb.DMatrix(X, feature_names=feat_names)
        return model.predict(dtest)
    except:
        return np.zeros(len(df))



def hybrid_rank(df, cb, cf, lf=None, xgb_s=None,
                w_cb=0.2, w_cf=0.2, w_lf=0.3, w_xgb=0.3):
    out = df.copy()
    out['cb_score']  = cb
    out['cf_score']  = cf
    out['lf_score']  = lf  if lf  is not None else 0.0
    out['xgb_score'] = xgb_s if xgb_s is not None else 0.0
    out['final_score'] = (
        w_cb*out['cb_score'] + w_cf*out['cf_score'] +
        w_lf*out['lf_score'] + w_xgb*out['xgb_score']
    )
    return out.sort_values('final_score', ascending=False)


from sklearn.metrics import ndcg_score

def precision_at_k(y_true, y_score, k=10):
    top_k = np.argsort(y_score)[::-1][:k]
    return np.mean([y_true[i] for i in top_k])

def recall_at_k(y_true, y_score, k=10):
    top_k = np.argsort(y_score)[::-1][:k]
    return np.sum([y_true[i] for i in top_k]) / np.sum(y_true) if np.sum(y_true) > 0 else 0

def ndcg_at_k(y_true, y_score, k=10):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_score = y_score.reshape(1, -1)
    return ndcg_score(y_true, y_score, k=k)

def evaluate_model(y_true, y_score, k=10):
    return {
        'precision@%d' % k: round(precision_at_k(y_true, y_score, k), 4),
        'recall@%d' % k: round(recall_at_k(y_true, y_score, k), 4),
        'ndcg@%d' % k: round(ndcg_at_k(y_true, y_score, k), 4),
    }
