import recommendation

print("\n[INFO] Starting training explicitly after reload...")
df_shape = recommendation.pd.read_csv(recommendation.INTERACTIONS_F).shape
print(f"[Explicit reload check] interactions.csv shape: {df_shape}")

print("\n[STEP 1] Training Collaborative Filtering...")
cf = recommendation.train_and_eval_cf()
print("[STEP 1 Done] CF =", cf is not None)

print("\n[STEP 2] Training LightFM model...")
lf = recommendation.train_lightfm()
print("[STEP 2 Done] LightFM =", lf is not None)

print("\n[STEP 3] Training XGBoost model...")
xgb = recommendation.train_xgb_ranker()
print("[STEP 3 Done] XGBoost =", xgb is not None)

print("\n[INFO] Final Summary:")
print("CF model trained:", cf is not None)
print("LightFM model trained:", lf is not None)
print("XGB model trained:", xgb is not None)
