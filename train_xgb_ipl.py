import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# === Load dataset ===
df = pd.read_csv("ipl_broadcast_viewership.csv")   # <-- Change filename if needed

print("✅ Columns detected in dataset:\n", list(df.columns))

# === Target variable ===
target = "ad_inventory_price_lakhs"
if target not in df.columns:
    raise ValueError(f"❌ Target column '{target}' not found!")

# === Handle missing values ===
df = df.dropna(subset=[target])
df = df.fillna(0)

# === Split features and target ===
X = df.drop(columns=[target])
y = df[target]

# === Encode categorical columns ===
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

print("✅ Encoded categorical features successfully.")

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Train XGBoost Regressor ===
model = xgb.XGBRegressor(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    enable_categorical=False
)

print("🚀 Training model on all dataset features...")
model.fit(X_train, y_train)

# === Predictions & Metrics ===
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Evaluation:")
print(f"  MAE: {mae:.3f}")
print(f"  MSE: {mse:.3f}")
print(f"  R²: {r2:.3f}")

# === Save model & encoders ===
joblib.dump({
    "model": model,
    "label_encoders": label_encoders,
    "features": list(X.columns)
}, "ipl_model.pkl")

print("💾 Model saved successfully as 'ipl_model.pkl'")

# === Feature Importance Plot ===
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, ax=plt.gca(), importance_type='gain', title='Feature Importance')
plt.tight_layout()
plt.show()
