import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import matplotlib.pyplot as plt

# ==========================
#  CẤU HÌNH GIAO DIỆN
# ==========================
st.set_page_config(page_title="Dự đoán giá xe máy", layout="centered")

# ==========================
#  LOAD DATA CHO CHART + DROPDOWN
# ==========================
DATA_PATH = "motorbike_cleaned.csv"
df = None
if os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception:
        df = None

# ==========================
#  ẢNH BANNER
# ==========================
if os.path.exists("xe_may_cu.jpg"):
    st.image("xe_may_cu.jpg", use_container_width=True)

# ==========================
#  BIỂU ĐỒ TOP 5 MODEL
# ==========================
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
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Dòng xe")
    ax.set_ylabel("Số lượng tin rao")
    ax.tick_params(axis="x", rotation=20)

    st.pyplot(fig)

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
#  LOAD MODEL
# ==========================
import joblib

MODEL_PATH = "motobike_price_model.joblib"
model = joblib.load(MODEL_PATH)



# Lúc train model dùng các feature này:
expected_features = ["mileage", "years_used", "model", "category"]
numeric_features = ["mileage", "years_used"]
categorical_features = ["model", "category"]
# engine_capacity chỉ hiển thị trên UI, không đưa vào model
# trừ khi sau này đạo hữu retrain pipeline có thêm cột này.

# ==========================
#  DROPDOWN OPTIONS TỪ DATA
# ==========================
select_options = {}
for col in categorical_features:
    if df is not None and col in df.columns:
        vals = sorted(df[col].dropna().astype(str).unique().tolist())
        if col == "model":
            # model: chỉ (Không chọn) + danh sách → vì mình cho thêm ô "tự nhập"
            select_options[col] = ["(Không chọn)"] + vals
        else:
            # category: giữ thêm "Khác..."
            select_options[col] = ["(Không chọn)"] + vals + ["Khác..."]
    else:
        select_options[col] = ["(Không chọn)"]

# =====================================================
#  BOX 1: DỰ ĐOÁN GIÁ XE MÁY – NGƯỜI MUA
# =====================================================
st.markdown("## 🚀 Dự đoán giá xe máy – Người mua")
st.subheader("📘 Nhập thông tin xe để dự đoán")

with st.form("form_du_doan"):
    # --- Numeric: mileage, years_used, engine_capacity ---
    c1, c2, c3 = st.columns(3)
    mileage = c1.text_input("Số km đã đi:", "15000")
    years_used = c2.text_input("Số năm sử dụng:", "2")
    engine_capacity = c3.text_input("Phân khối (cc):", "125")  # chỉ hiển thị, chưa đưa vào model

    # --- Categorical: model ---
    model_sel = st.selectbox("Dòng xe (model):", select_options["model"])
    model_free = st.text_input("Hoặc tự nhập dòng xe:", "")

    if model_free.strip():
        model_input = model_free.strip()
    elif model_sel == "(Không chọn)":
        model_input = np.nan
    else:
        model_input = model_sel

    # --- Categorical: category ---
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
            # engine_capacity hiện tại không đưa vào model
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

# =====================================================
#  BOX 2: PHÁT HIỆN GIÁ ĐĂNG BÁN BẤT THƯỜNG – NGƯỜI BÁN
# =====================================================
st.markdown("---")
st.markdown("## 🧭 Phát hiện giá đăng bán bất thường – Người bán")
st.subheader("📦 Kiểm tra mức giá bạn định đăng")

with st.form("form_phat_hien"):
    # --- Numeric ---
    c1s, c2s, c3s = st.columns(3)
    mileage_s = c1s.text_input("Số km đã đi:", "15000", key="seller_mileage")
    years_used_s = c2s.text_input("Số năm sử dụng:", "2", key="seller_years")
    engine_capacity_s = c3s.text_input("Phân khối (cc):", "125", key="seller_cc")  # chỉ hiển thị

    # --- Categorical: model ---
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

    # --- Categorical: category ---
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

    price_s = st.text_input("Giá bạn muốn đăng (VND):", "20000000", key="seller_price")

    submit_sell = st.form_submit_button("🧮 Kiểm tra giá có hợp lý không")

if submit_sell:
    X_sell = pd.DataFrame(
        [{
            "mileage": to_number_from_str(mileage_s),
            "years_used": to_number_from_str(years_used_s),
            # engine_capacity_s chưa đưa vào model
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

