import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# ==========================
#  CẤU HÌNH GIAO DIỆN + MENU
# ==========================
st.set_page_config(page_title="Dự đoán giá xe máy cũ", layout="centered")

st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Chọn mục:",
    [
        "Tên thành viên",
        "Tóm tắt dự án",
        "Xây dựng mô hình",
        "Dự đoán giá",
        "Xác định xe bất thường",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Nhóm thực hiện")
st.sidebar.markdown(
    """
- **HV1:** Phạm Văn Hải – email: haipham2403@gmail.com  
- **HV2:** Nguyễn Trần Xuân Linh – email: xuanlinh86@gmail.com  
"""
)

DATA_PATH = "motorbike_cleaned.csv"


# ==========================
#  LOAD DATA
# ==========================
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            return None
    return None


df = load_data()


# ==========================
#  HÀM PHỤ TRỢ
# ==========================
def to_number_from_str(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s)
    s = re.sub(r"[^\d]", "", s)
    return float(s) if s else np.nan


def format_vnd(x):
    try:
        x = float(x)
        return f"{int(x):,} ₫".replace(",", ".")
    except Exception:
        return str(x)


# ==========================
#  TRAIN MODEL TRỰC TIẾP TỪ CSV
# ==========================
@st.cache_resource
def train_model():
    if df is None:
        raise ValueError("Không đọc được file motorbike_cleaned.csv để train mô hình.")

    data = df.copy()

    # ---- Xử lý lại cột price giống script train ----
    if all(col in data.columns for col in ["price_min", "price_max", "price"]):
        data["mid_price"] = (data["price_min"] + data["price_max"]) / 2
        cond_outside = (data["price"] < data["price_min"]) | (data["price"] > data["price_max"])
        data.loc[cond_outside, "price"] = data.loc[cond_outside, "mid_price"]

    # ---- Làm sạch mileage, years_used ----
    def clean_numeric(col):
        return (
            col.astype(str)
               .str.replace(r"[^0-9\.\-]", "", regex=True)
               .replace("", np.nan)
               .astype(float)
        )

    if "mileage" in data.columns:
        data["mileage"] = clean_numeric(data["mileage"])
    else:
        data["mileage"] = np.nan

    if "years_used" in data.columns:
        data["years_used"] = clean_numeric(data["years_used"])
    else:
        data["years_used"] = np.nan

    # Bỏ dòng thiếu dữ liệu quan trọng
    data = data.dropna(subset=["price", "mileage", "years_used"])

    feature_numeric = ["mileage", "years_used"]
    feature_cat = [c for c in ["model", "category"] if c in data.columns]

    X = data[feature_numeric + feature_cat]
    y = data["price"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cat),
        ]
    )

    gbr = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", gbr),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"[MODEL TRAIN] RMSE: {rmse:,.0f}")
    print(f"[MODEL TRAIN] R2:   {r2:.4f}")

    expected_features = ["mileage", "years_used", "model", "category"]
    return pipeline, expected_features


model, expected_features = train_model()
numeric_features = ["mileage", "years_used"]
categorical_features = ["model", "category"]


# ==========================
#  ẢNH + CHART (DÙNG CHUNG)
# ==========================
def show_banner_and_chart():
    # Banner
    if os.path.exists("xe_may_cu.jpg"):
        st.image("xe_may_cu.jpg", use_container_width=True)

    # Biểu đồ top 5 model
    if df is not None and "model" in df.columns:
        st.subheader("📊 Các dòng xe phổ biến nhất trên thị trường (Top 5)")

        top5 = (
            df["model"]
            .dropna()
            .astype(str)
            .value_counts()
            .head(5)
            .reset_index()
        )
        top5.columns = ["model", "count"]

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#1A73E8", "#9B59B6"]

        ax.bar(top5["model"], top5["count"], color=colors[: len(top5)])

        for i, v in enumerate(top5["count"]):
            ax.text(
                i,
                v,
                str(v),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("Dòng xe")
        ax.set_ylabel("Số lượng tin rao")
        ax.tick_params(axis="x", rotation=20)

        st.pyplot(fig)


# ==========================
#  DROPDOWN OPTIONS
# ==========================
select_options = {}
for col in categorical_features:
    if df is not None and col in df.columns:
        vals = sorted(df[col].dropna().astype(str).unique().tolist())
        if col == "model":
            select_options[col] = ["(Không chọn)"] + vals
        else:
            select_options[col] = ["(Không chọn)"] + vals + ["Khác..."]
    else:
        select_options[col] = ["(Không chọn)"]


# ==========================
#  PAGE: DỰ ĐOÁN GIÁ – NGƯỜI MUA
# ==========================
def page_du_doan():
    st.markdown("## 🚀 Dự đoán giá xe máy – Người mua")
    st.subheader("📘 Nhập thông tin xe để dự đoán")

    with st.form("form_du_doan"):
        c1, c2, c3 = st.columns(3)
        mileage = c1.text_input("Số km đã đi:", "15000")
        years_used = c2.text_input("Số năm sử dụng:", "2")
        engine_capacity = c3.text_input("Phân khối (cc):", "125")  # chỉ hiển thị

        model_sel = st.selectbox("Dòng xe (model):", select_options["model"])
        model_free = st.text_input("Hoặc tự nhập dòng xe:", "")

        if model_free.strip():
            model_input = model_free.strip()
        elif model_sel == "(Không chọn)":
            model_input = np.nan
        else:
            model_input = model_sel

        category_sel = st.selectbox("Loại xe (category):", select_options["category"])
        if category_sel == "Khác...":
            category_input = st.text_input("Nhập loại xe khác:")
        elif category_sel == "(Không chọn)":
            category_input = np.nan
        else:
            category_input = category_sel

        submit_buy = st.form_submit_button("🔍 Dự đoán giá")

    if submit_buy:
        X_buy = pd.DataFrame(
            [{
                "mileage": to_number_from_str(mileage),
                "years_used": to_number_from_str(years_used),
                "model": model_input,
                "category": category_input,
            }]
        ).reindex(columns=expected_features)

        st.write("### Dữ liệu gửi vào mô hình (người mua)")
        st.dataframe(X_buy)

        try:
            y_pred = float(model.predict(X_buy)[0])
            st.success("🎯 Dự đoán thành công!")
            st.metric("Giá dự đoán (tham khảo cho người mua)", format_vnd(y_pred))
        except Exception as e:
            st.error("Lỗi khi dự đoán (người mua).")
            st.exception(e)


# ==========================
#  PAGE: PHÁT HIỆN GIÁ BẤT THƯỜNG – NGƯỜI BÁN
# ==========================
def page_phat_hien():
    st.markdown("## 🧭 Phát hiện giá đăng bán bất thường – Người bán")
    st.subheader("📦 Kiểm tra mức giá bạn định đăng")

    with st.form("form_phat_hien"):
        c1s, c2s, c3s = st.columns(3)
        mileage_s = c1s.text_input("Số km đã đi:", "15000", key="seller_mileage")
        years_used_s = c2s.text_input("Số năm sử dụng:", "2", key="seller_years")
        engine_capacity_s = c3s.text_input("Phân khối (cc):", "125", key="seller_cc")

        model_sel_s = st.selectbox(
            "Dòng xe (model):", select_options["model"], key="seller_model_sel"
        )
        model_free_s = st.text_input(
            "Hoặc tự nhập dòng xe (người bán):", "", key="seller_model_free"
        )

        if model_free_s.strip():
            model_input_s = model_free_s.strip()
        elif model_sel_s == "(Không chọn)":
            model_input_s = np.nan
        else:
            model_input_s = model_sel_s

        category_sel_s = st.selectbox(
            "Loại xe (category):", select_options["category"], key="seller_cat_sel"
        )
        if category_sel_s == "Khác...":
            category_input_s = st.text_input(
                "Nhập loại xe khác:", key="seller_cat_other"
            )
        elif category_sel_s == "(Không chọn)":
            category_input_s = np.nan
        else:
            category_input_s = category_sel_s

        price_s = st.text_input(
            "Giá bạn muốn đăng (VND):", "20000000", key="seller_price"
        )

        submit_sell = st.form_submit_button("🧮 Kiểm tra giá có hợp lý không")

    if submit_sell:
        X_sell = pd.DataFrame(
            [{
                "mileage": to_number_from_str(mileage_s),
                "years_used": to_number_from_str(years_used_s),
                "model": model_input_s,
                "category": category_input_s,
            }]
        ).reindex(columns=expected_features)

        seller_price = to_number_from_str(price_s)

        st.write("### Dữ liệu gửi vào mô hình (người bán)")
        st.dataframe(X_sell)

        if np.isnan(seller_price):
            st.error("Vui lòng nhập 'Giá bạn muốn đăng' là số hợp lệ.")
        else:
            try:
                fair_price = float(model.predict(X_sell)[0])

                st.write("### Kết quả đánh giá giá đăng bán")
                st.write(f"- Giá hợp lý theo mô hình: **{format_vnd(fair_price)}**")
                st.write(f"- Giá bạn muốn đăng: **{format_vnd(seller_price)}**")

                if fair_price <= 0:
                    st.warning(
                        "Giá dự đoán không hợp lệ (<=0). Kiểm tra lại dữ liệu đầu vào hoặc mô hình."
                    )
                else:
                    ratio = seller_price / fair_price
                    low_ok = 0.9 * fair_price
                    high_ok = 1.1 * fair_price

                    if ratio < 0.7:
                        st.error(
                            "🚨 Giá **quá rẻ** so với mặt bằng dự đoán → có thể là tin bất thường hoặc bạn đang bán lỗ rất mạnh."
                        )
                    elif 0.7 <= ratio < 0.9:
                        st.warning(
                            "⚠️ Giá **thấp hơn thị trường**. Người mua rất có lợi, bạn nên cân nhắc lại mức giá."
                        )
                    elif 0.9 <= ratio <= 1.1:
                        st.success("✅ Giá **hợp lý**, nằm trong khoảng thị trường dự đoán.")
                    elif 1.1 < ratio <= 1.3:
                        st.info(
                            "ℹ️ Giá **hơi cao hơn** so với thị trường. Người mua có thể còn mặc cả."
                        )
                    else:
                        st.error(
                            "🚨 Giá **quá cao** so với thị trường → dễ bị xem là tin đăng không hấp dẫn hoặc bất thường."
                        )

                    st.write(
                        f"👉 Khoảng giá tham khảo nên đăng: **{format_vnd(low_ok)} – {format_vnd(high_ok)}**"
                    )

            except Exception as e:
                st.error("Lỗi khi đánh giá giá đăng bán.")
                st.exception(e)


# ==========================
#  ĐIỀU HƯỚNG THEO MENU
# ==========================
if menu == "Tên thành viên":
    st.title("👥 Thành viên thực hiện")
    st.markdown(
        """
**Đề tài:** Dự đoán giá xe máy cũ & phát hiện tin đăng bất thường trên Chợ Tốt.

**Nhóm thực hiện:**
- **Phạm Văn Hải** – xây dựng mô hình mô hình phát hiện bất thường.  
- **Nguyễn Trần Xuân Linh** – xây dựng mô hình dự báo giá.
"""
    )

elif menu == "Tóm tắt dự án":
    st.title("📌 Tóm tắt dự án")
    show_banner_and_chart()
    st.markdown(
        """
**Mục tiêu chính:**
- Xây dựng mô hình dự đoán giá xe máy cũ từ dữ liệu thu thập trên Chợ Tốt.
- Hỗ trợ **người mua** ước lượng giá tham khảo để tránh mua hớ.
- Hỗ trợ **người bán** kiểm tra xem mức giá đăng có bất thường (quá thấp / quá cao) hay không.

**Dữ liệu sử dụng:**
- Tin đăng xe máy cũ trên Chợ Tốt.
- Các thông tin chính gồm: giá, khoảng giá min–max, số km đã đi (mileage),
  số năm sử dụng (years_used), dòng xe (model), loại xe (category), v.v.

**Ý nghĩa ứng dụng:**
- Giúp sinh viên thực hành quy trình đầy đủ: thu thập dữ liệu – tiền xử lý –
  xây dựng mô hình máy học – triển khai thành web app thực tế.
"""
    )

elif menu == "Xây dựng mô hình":
    st.title("🧠 Xây dựng mô hình")
    st.markdown(
        """
Quy trình xây dựng mô hình gồm các bước:

### 1. Tiền xử lý dữ liệu
- Chuẩn hoá lại cột **price** dựa trên `price_min` và `price_max`.
- Làm sạch và chuyển kiểu dữ liệu cho:
  - `mileage` – số km đã đi  
  - `years_used` – số năm sử dụng  
- Loại bỏ các bản ghi thiếu dữ liệu quan trọng.

### 2. Chọn biến đưa vào mô hình
- Biến số (numeric):
  - `mileage`, `years_used`
- Biến phân loại (categorical):
  - `model`, `category`
- Dùng **OneHotEncoder(handle_unknown="ignore")** để mã hoá biến phân loại.

### 3. Các mô hình đã thử nghiệm
- Linear Regression (mô hình cơ bản để so sánh).
- Random Forest Regressor.
- Gradient Boosting Regressor.
- XGBoost.

### 4. Mô hình triển khai trên app
- Sử dụng **GradientBoostingRegressor**:
  - `n_estimators = 150`
  - `learning_rate = 0.1`
  - `max_depth = 3`
- Pipeline:
  - `ColumnTransformer (num + cat)` → `GradientBoostingRegressor`
- Chia dữ liệu train/test 70/30, đánh giá bằng:
  - **RMSE** (Root Mean Squared Error)
  - **R²** (hệ số xác định)

Mô hình sau khi huấn luyện được dùng trực tiếp trong app để:
- Dự đoán giá tham khảo cho **người mua**.
- Đưa ra khuyến nghị và cảnh báo giá bất thường cho **người bán**.
"""
    )

elif menu == "Dự đoán giá":
    show_banner_and_chart()
    page_du_doan()

elif menu == "Xác định xe bất thường":
    show_banner_and_chart()
    page_phat_hien()

