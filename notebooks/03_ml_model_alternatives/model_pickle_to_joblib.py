import pickle
import joblib

# %%
home_path = "/home/marcos/Documentos/comunidade_DS/pa004_health_insurance_cross_sell/"

# %%
bal_rf_tuned = pickle.load(open(home_path + 'ml_models_comp/giant/bal_rf_validated.pkl', 'rb'))

# %%
joblib.dump(bal_rf_tuned, open(home_path + 'ml_models_comp/giant/bal_rf_validated_3.joblib', 'wb'), compress=('lzma', 3))

# %%
joblib.dump(bal_rf_tuned, open(home_path + 'ml_models_comp/giant/bal_rf_validated_6.joblib', 'wb'), compress=('lzma', 6))

# %%
joblib.dump(bal_rf_tuned, open(home_path + 'ml_models_comp/giant/bal_rf_validated_9.joblib', 'wb'), compress=('lzma', 9))
