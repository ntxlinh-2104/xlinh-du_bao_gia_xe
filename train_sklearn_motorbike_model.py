import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# ================================
# 1. Đọc dữ liệu gốc
# ================================
# Đổi lại path nếu cần
CSV_PATH = r"D:/DATA_SCIENCE_KHTN/mon_7/project_1/motorbike_cleaned.csv"

df = pd.read_csv(CSV_PATH)

# ================================
# 2. Xử lý lại cột price giống Spark
# ================================
# Tạo mid_price = (price_min + price_max)/2
df["mid_price"] = (df["price_min"] + df["price_max"]) / 2

# Nếu price nằm ngoài khoảng [price_min, price_max] => thay bằng mid_price
cond_outside = (df["price"] < df["price_min"]) | (df["price"] > df["price_max"])
df.loc[cond_outside, "price"] = df.loc[cond_outside, "mid_price"]

# ================================
# 3. Làm sạch mileage, years_used giống Spark
# ================================
def clean_numeric(col):
    return (
        col.astype(str)
           .str.replace(r"[^0-9\.\-]", "", regex=True)
           .replace("", np.nan)
           .astype(float)
    )

if "mileage" in df.columns:
    df["mileage"] = clean_numeric(df["mileage"])
else:
    df["mileage"] = np.nan

if "years_used" in df.columns:
    df["years_used"] = clean_numeric(df["years_used"])
else:
    df["years_used"] = np.nan

# Bỏ các dòng thiếu giá trị quan trọng
df = df.dropna(subset=["price", "mileage", "years_used"])

# ================================
# 4. Chọn biến đầu vào / đầu ra
# ================================
feature_numeric = ["mileage", "years_used"]
feature_cat = []
if "model" in df.columns:
    feature_cat.append("model")
if "category" in df.columns:
    feature_cat.append("category")

X = df[feature_numeric + feature_cat]
y = df["price"]

# ================================
# 5. Pipeline sklearn: OneHotEncoder + GBR
# ================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", feature_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cat),
    ]
)

gbr = GradientBoostingRegressor(
    n_estimators=150,    # gần tương đương maxIter=150 trong Spark
    learning_rate=0.1,   # giống stepSize=0.1
    max_depth=3,         # độ sâu cây con
    random_state=42
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", gbr),
    ]
)

# ================================
# 6. Train / test split (để tham khảo)
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model.fit(X_train, y_train)

# Đánh giá nhanh
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:,.0f}")
print(f"R2:   {r2:.4f}")

# ================================
# 7. Lưu pipeline sklearn sang file .pkl
# ================================
PKL_NAME = "motobike_price_model.pkl"  # đúng tên đạo hữu yêu cầu

with open(PKL_NAME, "wb") as f:
    pickle.dump(model, f)

print(f"\n✅ Đã lưu mô hình sklearn vào file: {PKL_NAME}")
print("   (File này chứa FULL pipeline: preprocess + model)")
