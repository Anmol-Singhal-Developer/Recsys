import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join('static', 'charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_charts():
    plt.style.use('ggplot') 

    df1 = pd.read_csv(os.path.join(DATA_DIR, 'search_history.csv'))
    top_loc = df1['locality'].value_counts().nlargest(4)
    plt.figure(figsize=(6, 4))
    top_loc.plot(kind='bar', color='#5DADE2')
    plt.title('Top 4 Localities Searched', fontsize=14)
    plt.ylabel('Searches')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_localities.png'))
    plt.close()

    df2 = pd.read_csv(os.path.join(DATA_DIR, 'recipe_history.csv'))
    top_rec = df2['name'].value_counts().nlargest(4)
    plt.figure(figsize=(6, 4))
    top_rec.plot(kind='bar', color='#82E0AA')
    plt.title('Top 4 Recipes Recommended', fontsize=14)
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_recipes.png'))
    plt.close()

    df3 = pd.read_json(os.path.join(DATA_DIR, 'user_uploaded_recipes.json'))
    df3['Cuisine'] = df3['Cuisine'].astype(str).str.replace(r"[\[\]'\"]", "", regex=True).str.strip()
    top_cuisine = df3['Cuisine'].value_counts().nlargest(4)
    plt.figure(figsize=(6, 4))
    top_cuisine.plot(kind='bar', color='#F1948A')
    plt.title('Top 4 Uploaded Cuisines', fontsize=14)
    plt.ylabel('Uploads')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_cuisines.png'))
    plt.close()
