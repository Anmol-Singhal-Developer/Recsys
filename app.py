import os
import json
import ast
import warnings
from datetime import datetime
import csv

from flask import (
    Flask, render_template, request,
    redirect, url_for, session, jsonify, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from joblib import load
import xgboost as xgb
import numpy as np
import pandas as pd
from visualization import generate_charts


warnings.filterwarnings(
    "ignore",
    message="LightFM was compiled without OpenMP support"
)

from recommendation import (
    load_restaurant_df,
    get_user_profile, update_user_profile,
    CF_MODEL_F, LF_MODEL_F, XGB_MODEL_F,
    score_cf_surprise, score_lightfm, score_xgb,
    prepare_content_features, score_content, hybrid_rank,
    haversine_distance, USER_PROFILES_F, INTERACTIONS_F
)

app = Flask(__name__)
app.secret_key = 'hello'

BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RADIUS_KM  = 10

cf_model  = load(CF_MODEL_F) if os.path.exists(CF_MODEL_F) else None
lf_model  = load(LF_MODEL_F) if os.path.exists(LF_MODEL_F) else None
xgb_model = xgb.Booster()
if os.path.exists(XGB_MODEL_F):
    xgb_model.load_model(XGB_MODEL_F)

def load_user_profiles():
    if os.path.exists(USER_PROFILES_F):
        return json.load(open(USER_PROFILES_F))
    return {}
def save_user_profiles(P):
    json.dump(P, open(USER_PROFILES_F,'w'), indent=2)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method=='POST':
        f = request.form
        first, last = f['first_name'].strip(), f['last_name'].strip()
        u = f['username'].strip(); pwd, conf = f['password'], f['confirm_password']
        age = f['age'].strip()
        if not all([first,last,u,pwd,conf,age]):
            flash("All fields required","error"); return redirect(url_for('register'))
        if pwd!=conf:
            flash("Passwords must match","error"); return redirect(url_for('register'))
        if not age.isdigit() or int(age)<13:
            flash("Age ≥13 required","error"); return redirect(url_for('register'))
        P = load_user_profiles()
        if u in P:
            flash("Username taken","error"); return redirect(url_for('register'))
        P[u] = {
            'first_name': first,
            'last_name' : last,
            'password_hash': generate_password_hash(pwd),
            'age': int(age)
        }
        save_user_profiles(P)
        flash("Registered! Please log in.","success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        u,p = request.form['username'].strip(), request.form['password']
        P = load_user_profiles(); user = P.get(u)
        if user and check_password_hash(user['password_hash'], p):
            session['username'] = u
            flash(f"Welcome back, {user['first_name']}!","success")
            return redirect(url_for('set_survey'))
        flash("Invalid credentials","error")
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/set_survey', methods=['GET','POST'])
def set_survey():
    if 'username' not in session:
        return redirect(url_for('login'))
    df = load_restaurant_df()
    cuisines  = sorted({c for row in df['CuisineList'] for c in row})
    amenities = sorted(col.replace('Highlight_','')
                       for col in df.columns if col.startswith('Highlight_'))
    min_b = int(df['AverageCostForTwo'].min()//2)
    max_b = int(df['AverageCostForTwo'].max()//2)
    if request.method=='POST':
        c = request.form.getlist('cuisines')
        a = request.form.getlist('amenities')
        b = int(request.form['budget'])
        update_user_profile(session['username'], cuisines=c, amenities=a, budget=b)
        return redirect(url_for('set_location'))
    return render_template('survey.html',
        cuisines=cuisines, amenities=amenities,
        min_budget=min_b, max_budget=max_b
    )

@app.route('/set_location', methods=['GET','POST'])
def set_location():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method=='POST':
        lat, lon = float(request.form['lat']), float(request.form['lon'])
        update_user_profile(session['username'], home_lat=lat, home_lon=lon)
        return redirect(url_for('preferences'))
    return render_template('set_location.html')

@app.route('/api/localities')
def api_localities():
    city = request.args.get('city','')
    df = load_restaurant_df()
    if city:
        df = df[df['City']==city]
    return jsonify(sorted(df['Locality'].dropna().unique()))

@app.route('/api/options')
def api_options():
    city,loc = request.args.get('city',''), request.args.get('locality','')
    df = load_restaurant_df()
    if city: df = df[df['City']==city]
    if loc:  df = df[df['Locality']==loc]
    cuisines  = sorted({c for row in df['CuisineList'] for c in row})
    amenities = sorted(col.replace('Highlight_','')
                       for col in df.columns
                       if col.startswith('Highlight_') and df[col].sum()>0)
    return jsonify({
        'cuisines': cuisines,
        'amenities': amenities,
        'min_budget': int(df['AverageCostForTwo'].min()//2),
        'max_budget': int(df['AverageCostForTwo'].max()//2)
    })

@app.route('/preferences', methods=['GET','POST'])
def preferences():
    if 'username' not in session:
        return redirect(url_for('login'))
    u = session['username']
    prof = get_user_profile(u) or {}
    df = load_restaurant_df()

    if {'Latitude','Longitude'}.issubset(df.columns):
        lat, lon = prof.get('home_lat'), prof.get('home_lon')
        if lat is not None and lon is not None:
            df['distance_km'] = df.apply(
                lambda r: haversine_distance(lon, lat, r['Longitude'], r['Latitude']),
                axis=1
            )
            df = df[df['distance_km'] <= RADIUS_KM]

    cities = sorted(df['City'].dropna().unique())

    if request.method == 'POST':
        f    = request.form
        city = f['city']; loc = f['locality']
        c    = f.getlist('cuisines')
        a    = f.getlist('amenities')
        b    = int(f['budget'])

        sh = os.path.join(DATA_DIR, 'search_history.csv')
        if not os.path.exists(sh):
            open(sh, 'w').write('username,city,locality,cuisines,amenities,budget_pp,timestamp\n')
        with open(sh,'a') as out:
            out.write(','.join([
                u, city, loc,
                ';'.join(c), ';'.join(a),
                str(b), datetime.utcnow().isoformat()
            ]) + '\n')

        df1 = df[(df['City']==city) & (df['Locality']==loc)]
        df1 = df1[df1['AverageCostForTwo']/2 <= b]
        if c:
            df1 = df1[df1['CuisineList'].apply(lambda L: any(x in L for x in c))]
        for tag in a:
            col = f'Highlight_{tag}'
            if col in df1.columns:
                df1 = df1[df1[col] == 1]
        if df1.empty:
            flash(f"No matches in {loc}—showing all","warning")
            df1 = df[(df['City']==city) & (df['Locality']==loc)]

        df_feat, X, scaler, _, feat_names = prepare_content_features(df1, b, b, 24)
        cb = score_content(df_feat, X, scaler, feat_names, c, a, b, b, 24)
        cf = score_cf_surprise(cf_model, df_feat, u)
        lf = score_lightfm(lf_model, df_feat, u)
        xg = score_xgb(xgb_model, df1)
        print("User selected cuisines:", c)
        print("Matched in df1:", [x for x in c if any(x in row for row in df1['CuisineList'])])


        ranked = hybrid_rank(df_feat, cb, cf, lf, xg).head(5).reset_index()

        records = []
        amen_cols = [col for col in df1.columns if col.startswith('Highlight_')]

        for _, row in ranked.iterrows():
            res_id = row['ResId']
            orig = df1[df1['ResId'] == res_id].iloc[0]

            amenities = [
                col.replace('Highlight_','')
                for col in amen_cols
                if orig[col] == 1
            ]
            dist = orig.get('distance_km')

            if dist is None:
                dist = 0.0
            records.append({
                'ResId'       : res_id,
                'Name'        : orig.get('Name',''),
                'Address'     : orig.get('Address',''),
                'CuisineList' : orig.get('CuisineList', []),
                'Amenities'   : amenities,
                'distance_km' : dist,
                'final_score': float(row.get('hybrid_score', 0.0))

            })

        return render_template(
            'recommendations.html',
            recs=records
        )


    return render_template('preferences.html', cities=cities)



@app.route('/record_feedback', methods=['POST'])
def record_feedback():
    u = session.get('username')
    if not u:
        flash("Please log in","error")
        return redirect(url_for('login'))
    likes   = request.form.getlist('likes')
    ratings = {
        k.split('_',1)[1]:int(v)
        for k,v in request.form.items()
        if k.startswith('rating_') and v.isdigit()
    }
    rows = [(u,rid, ratings.get(rid,1)) for rid in set(likes)|set(ratings)]
    if rows:
        df0 = pd.DataFrame(rows, columns=['user_id','ResId','interaction'])
        if os.path.exists(INTERACTIONS_F):
            hist = pd.read_csv(INTERACTIONS_F, dtype=str)
            all_ = pd.concat([hist,df0],ignore_index=True)
            all_.drop_duplicates(['user_id','ResId'],keep='last',inplace=True)
        else:
            all_ = df0
        all_.to_csv(INTERACTIONS_F,index=False)
        flash("Feedback saved","success")
    else:
        flash("No feedback","info")
    return redirect(url_for('preferences'))

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out","info")
    return redirect(url_for('login'))


GROCERY_F = os.path.join(BASE_DIR,'user_grocery.json')
def load_grocery():
    if os.path.exists(GROCERY_F):
        return json.load(open(GROCERY_F))
    return {}
def save_grocery(g):
    json.dump(g, open(GROCERY_F,'w'), indent=2)

@app.route('/grocery')
def grocery():
    u = session.get('username')
    if not u:
        return redirect(url_for('login'))
    data = load_grocery().get(u, [])
    return render_template('grocery_list.html', grocery_list=data)

@app.route('/grocery/add', methods=['POST'])
def grocery_add():
    u = session.get('username')
    it = request.form.get('item','').strip()
    if u and it:
        G = load_grocery(); L=G.get(u,[])
        if it not in L: L.append(it); G[u]=L; save_grocery(G)
    return redirect(url_for('grocery'))

@app.route('/grocery/remove', methods=['POST'])
def grocery_remove():
    u = session.get('username')
    it = request.form.get('item','').strip()
    G=load_grocery(); L=G.get(u,[])
    if it in L: L.remove(it); G[u]=L; save_grocery(G)
    return redirect(url_for('grocery'))


def parse_list_cell(x):
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return []
    try:
        parsed = ast.literal_eval(x)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except:
        return []


recipes_df = pd.read_csv(os.path.join(DATA_DIR,'Recipes_Featured.csv'))
FIELDNAMES = list(dict.fromkeys(recipes_df.columns.tolist()))
recipes_df.columns = [c.strip() for c in recipes_df.columns]
if 'Recipe Id' in recipes_df.columns:
    recipes_df.rename(columns={'Recipe Id':'RecipeID'}, inplace=True)
for c in ('Dietary','Equipment','Special'):
    recipes_df[f'{c}_list'] = recipes_df.get(c,'').fillna('').apply(parse_list_cell)

REC_CF = os.path.join(MODELS_DIR,'recipe_cf.joblib')
REC_LF = os.path.join(MODELS_DIR,'recipe_lightfm.joblib')
REC_XG = os.path.join(MODELS_DIR,'recipe_xgb_ranker.model')
svd_recipe = load(REC_CF) if os.path.exists(REC_CF) else None
lf_recipe  = load(REC_LF) if os.path.exists(REC_LF) else None
try:
    xgb_recipe = xgb.Booster()
    if os.path.exists(REC_XG):
        xgb_recipe.load_model(REC_XG)
    else:
        xgb_recipe = None
except:
    xgb_recipe = None

@app.route('/recipes', methods=['GET','POST'])
def recipes_home():
    cuisines   = sorted(recipes_df['Cuisine'].dropna().unique())
    categories = sorted(recipes_df['Recipe Category'].dropna().unique())
    dietary   = sorted({t
                    for tags in recipes_df['Dietary_list']
                    if isinstance(tags, (list, tuple))
                    for t in tags})
    equipment = sorted({t
                    for tags in recipes_df['Equipment_list']
                    if isinstance(tags, (list, tuple))
                    for t in tags})
    special   = sorted({t
                    for tags in recipes_df['Special_list']
                    if isinstance(tags, (list, tuple))
                    for t in tags})


    recs = None
    if request.method == 'POST':
        if 'username' not in session:
            return redirect(url_for('login'))
        user     = session['username']
        f        = request.form
        cuisine  = f.get('cuisine','')
        category = f.get('category','')
        min_r    = float(f.get('min_rating') or 0)
        diets    = f.getlist('dietary')
        equips   = f.getlist('equipment')
        specials = f.getlist('special')

        df = recipes_df.copy()
        if cuisine:   df = df[df['Cuisine']==cuisine]
        if category:  df = df[df['Recipe Category']==category]
        if min_r>0:   df = df[df['Rating']>=min_r]
        for tag in diets:
            df = df[df['Dietary_list'].apply(lambda tags: tag in tags)]
        for tag in equips:
            df = df[df['Equipment_list'].apply(lambda tags: tag in tags)]
        for tag in specials:
            df = df[df['Special_list'].apply(lambda tags: tag in tags)]

        if df.empty:
            flash("No matches—showing 5 random recipes","warning")
            df = recipes_df.sample(5)

        scored = []
        for _,r in df.iterrows():
            try:
                rating = float(r['Rating'])
            except (TypeError, ValueError):
                rating = 0.0
            score = rating
            # CF
            if svd_recipe:
                try:    score += svd_recipe.predict(user,str(r['RecipeID'])).est
                except: pass
            # LightFM
            if lf_recipe:
                try:
                    uid = lf_recipe.dataset.mapping()[0][user]
                    iid = lf_recipe.dataset.mapping()[2][str(r['RecipeID'])]
                    score += lf_recipe.predict(uid,[iid])[0]
                except: pass
            if xgb_recipe:
                nc = len(str(r.get('Cuisine', '')).split(',')) if pd.notna(r.get('Cuisine')) else 0
                nd = len(str(r.get('Dietary', '')).split(',')) if pd.notna(r.get('Dietary')) else 0
                arr = np.array([[rating, nc, nd]], dtype=float)

                dmat = xgb.DMatrix(
                arr,
                feature_names=['interaction', 'n_cuisine', 'n_dietary']
                )

                try:
                    score += float(xgb_recipe.predict(dmat)[0])
                except:
                    pass
            scored.append((score,r))

        scored.sort(key=lambda x:x[0], reverse=True)
        top5 = [rec.to_dict() for _,rec in scored[:5]]
        history_path = os.path.join(DATA_DIR, 'recipe_history.csv')
        is_new = not os.path.exists(history_path)

        with open(history_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(['user_id', 'recipe_id', 'name', 'cuisine', 'category', 'rating', 'timestamp'])

            for rec in top5:
                writer.writerow([
                    session['username'],
                        rec['RecipeID'],
                        rec['Name'],
                        rec['Cuisine'],
                        rec['Recipe Category'],
                        rec['Rating'],
                        datetime.utcnow().isoformat()
                    ])


        recs = []
        for rec in top5:
            rid = rec['RecipeID']
            recs.append({
                'id'            : rid,
                'name'          : rec['Name'],
                'cuisine': (', '.join(rec['Cuisine']) if isinstance(rec['Cuisine'], (list, tuple)) else str(rec['Cuisine']) if rec['Cuisine'] else ''),
                'category'      : rec['Recipe Category'],
                'rating'        : rec['Rating'],
                'review_count'  : rec.get('Review Count',0),
                'dietary_tags': list(rec['Dietary_list']) if isinstance(rec['Dietary_list'], (list, tuple)) else [],
                'equipment_tags': list(rec['Equipment_list']) if isinstance(rec['Equipment_list'], (list, tuple)) else [],
                'special_tags': list(rec['Special_list']) if isinstance(rec['Special_list'], (list, tuple)) else [],
                'ingredients': ast.literal_eval(rec['Ingredients']) if isinstance(rec['Ingredients'], str) else rec['Ingredients'],
                'steps': ast.literal_eval(rec['Steps']) if isinstance(rec['Steps'], str) else rec['Steps'],
                'prep_time'     : rec.get('Preptime',''),
                'total_time'    : rec.get('Totaltime','')
            })

    return render_template('recipes_home.html',
        cuisine_options   = cuisines,
        category_options  = categories,
        dietary_options   = dietary,
        equipment_options = equipment,
        special_options   = special,
        recommendations   = recs
    )

@app.route('/recipes/<int:recipe_id>/add_to_grocery', methods=['POST'])
def add_to_grocery(recipe_id):
    u = session.get('username')
    if not u:
        return redirect(url_for('login'))
    rec = recipes_df[recipes_df['RecipeID']==recipe_id]
    if not rec.empty:
        ings = rec.iloc[0]['Ingredients'].split('\n')
        g = load_grocery = lambda: json.load(open(os.path.join(BASE_DIR,'user_grocery.json'))) \
                               if os.path.exists(os.path.join(BASE_DIR,'user_grocery.json')) else {}
        G = g()
        lst = G.get(u, [])
        for i in ings:
            if i and i not in lst:
                lst.append(i)
        G[u] = lst
        json.dump(G, open(os.path.join(BASE_DIR,'user_grocery.json'),'w'), indent=2)
        flash("Ingredients added to your grocery list","success")
    return redirect(url_for('grocery'))

@app.route('/upload', methods=['GET','POST'])
def upload():

    cuisines      = sorted(recipes_df['Cuisine'].dropna().unique())
    categories    = sorted(recipes_df['Recipe Category'].dropna().unique())
    dietary_opts   = sorted({t
                    for tags in recipes_df['Dietary_list']
                    if isinstance(tags, (list, tuple))
                    for t in tags})
    equipment_opts = sorted({t
                    for tags in recipes_df['Equipment_list']
                    if isinstance(tags, (list, tuple))
                    for t in tags})
    special_opts   = sorted({t
                    for tags in recipes_df['Special_list']
                    if isinstance(tags, (list, tuple))
                    for t in tags})

    if request.method == 'GET':
        return render_template(
            'upload.html',
            cuisines=cuisines,
            categories=categories,
            dietary_opts=dietary_opts,
            equipment_opts=equipment_opts,
            special_opts=special_opts
        )


    f     = request.form
    name  = f.get('name','').strip()
    ings  = [l.strip() for l in f.get('ingredients','').splitlines() if l.strip()]
    steps = [l.strip() for l in f.get('steps','').splitlines()       if l.strip()]
    if not (name and ings and steps):
        flash("Name, ingredients & steps are required","danger")
        return redirect(url_for('upload'))

    def to_int(x):
        try: return int(x)
        except: return None
    def to_float(x):
        try: return float(x)
        except: return None

    parsed = {
      'RecipeID'              : int(recipes_df['RecipeID'].max() or 0) + 1,
      'Name'                  : name,
      'Cuisine'               : f.get('cuisine',''),
      'Recipe Category'       : f.get('category',''),
      'Rating'                : to_float(f.get('rating')),
      'Review Count'          : to_int(f.get('review_count')) or 0,
      'Calories'              : to_float(f.get('calories')),
      'Fat Content'           : to_float(f.get('fat_content')),
      'Saturated Fat Content' : to_float(f.get('sat_fat_content')),
      'Cholesterol Content'   : to_float(f.get('cholesterol')),
      'Sodium Content'        : to_float(f.get('sodium_content')),           
      'Carbohydrate Content'  : to_float(f.get('carb_content')),
      'Fiber Content'         : to_float(f.get('fiber_content')),
      'Sugar Content'         : to_float(f.get('sugar_content')),
      'Protein Content'       : to_float(f.get('protein_content')),
      'Recipe Servings'       : to_int(f.get('recipe_servings')) or 1,
      'Keywords Clean'        : f.get('keywords','').strip(),
      'Ingredients'           : "\n".join(ings),
      'Steps'                 : "\n".join(steps),
      'Cuisine'               : f.get('cuisine',''),
      'Dietary'               : str(f.getlist('dietary')),
      'Special'               : str(f.getlist('special')),
      'Equipment'             : str(f.getlist('equipment')),
      'Preptime'              : to_int(f.get('prep_time')),
      'Totaltime'             : to_int(f.get('total_time')),
      'num_ingredients'       : len(ings),
      'num_steps'             : len(steps),
    }


    row = {}
    for col in FIELDNAMES:
        row[col] = parsed.get(col, "")
        row['user_id'] = session.get('username', 'unknown')



    UPLOADS_JSON = os.path.join(DATA_DIR, 'user_uploaded_recipes.json')

    if os.path.exists(UPLOADS_JSON):
        with open(UPLOADS_JSON, 'r', encoding='utf-8') as f:
            uploaded_data = json.load(f)
    else:
        uploaded_data = []

    uploaded_data.append(row)

    with open(UPLOADS_JSON, 'w', encoding='utf-8') as f:
        json.dump(uploaded_data, f, indent=2)



    for c in ('Dietary', 'Equipment', 'Special'):
        recipes_df[f'{c}_list'] = recipes_df.get(c, '').fillna('').apply(parse_list_cell)


    flash("Recipe added!","success")
    return redirect(url_for('recipes_home'))


@app.route('/history')
def history():
    user = session.get('username')
    if not user:
        flash("Please login to view your history", "warning")
        return redirect(url_for('login'))


    rest_hist = pd.read_csv(os.path.join(DATA_DIR, 'search_history.csv'))
    rest_hist = rest_hist[rest_hist['user_id'] == user]


    recipe_hist = pd.read_csv(os.path.join(DATA_DIR, 'recipe_history.csv'))
    recipe_hist = recipe_hist[recipe_hist['user_id'] == user]

    uploaded = []
    uploaded_path = os.path.join(DATA_DIR, 'user_uploaded_recipes.json')
    if os.path.exists(uploaded_path):
        with open(uploaded_path) as f:
            all_recipes = json.load(f)
            uploaded = [r for r in all_recipes if r.get('user_id') == user]

    return render_template("history.html", restaurant_history=rest_hist.to_dict(orient='records'),
                           recipe_history=recipe_hist.to_dict(orient='records'),
                           uploaded_recipes=uploaded)

@app.route('/dashboard')
def dashboard():
    generate_charts()
    return render_template('dashboard.html')

if __name__=='__main__':
    app.run(debug=True)