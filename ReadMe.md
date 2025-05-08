# Food & Restaurant Recommendation System

This project is a smart food discovery platform that helps users get personalized suggestions for both **recipes** and **restaurants**. It combines machine learning models with user interaction history and content data to give meaningful food recommendations. It also supports recipe uploads, grocery list creation, and shows nutrition info.

---

## What’s in the Project?

- **Dual Recommendation Engine**:
  - Restaurant suggestions using user history, cuisine, rating, price, and location.
  - Recipe suggestions using food.com data, user preferences, and interaction logs.
- **Machine Learning Models**:
  - Collaborative Filtering (SVD)
  - LightFM
  - XGBoost Ranker
- **Hybrid Model**: Combines all three models for better results.
- 🛍️ Grocery planner + nutrition module.
- 📦 Upload your own recipes (stored for future use).
- 🖥️ Web interface built with Flask, Bootstrap, and HTML.

---

## Install Required Libraries
run pip install -r requirements.txt in anaconda prompt in project directory

## Pre Process Data
- Go to the Data folder, Open pre process files, Run all the three codes in the code folder.
- Go to final data column, run the EDA and Feature engineering code.
- Copy the generated final 3 csv file in the main data path.
- All of this is alreayd done and kept. 

### Trian the model
- Restaurant recommendation:
    run python train_models.py in anaconda prompt in project directory
- Recipe recommendation: 
    python train_recipe_models.py in anaconda prompt in project directory
 
### Run the Web app
- python app.py in anaconda prompt in project directory



