# du_doan_gia_xe.py
# ==========================
# ỨNG DỤNG STREAMLIT DỰ ĐOÁN GIÁ XE MÁY CŨ
# - Người mua: Dự đoán giá tham khảo
# - Người bán: Định giá & phát hiện xe bất thường
# - Quản trị viên: Duyệt / từ chối tin đăng, gửi lý do
# ==========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# ==========================
# KHỞI TẠO SESSION STATE
# ==========================

# Hàng chờ tin đăng bất thường cho quản trị viên
if "pending_posts" not in st.session_state:
    st.session_state["pending_posts"] = []

# ==========================
# HÀM LOAD MODEL & DATA
# ==========================

@st.cache_resource
def load_model():
    """
    Load pipeline sklearn đã train sẵn.
    Pipeline này phải nhận DataFrame và .predict trả ra giá (VND).

    👉 TODO: đổi path/file cho đúng với project của bạn:
        models/motorbike_price_pipeline.pkl
    """
    model_path = Path("models/motorbike_price_pipeline.pkl")
    if not model_path.exists():
        st.error("❌ Không tìm thấy file models/motorbike_price_pipeline.pkl\n"
                 "Hãy kiểm tra lại thư mục models và tên file model.")
        st.stop()
    model = joblib.load(model_path)
    return model


@st.cache_data
def load_data():
    """
    Load dữ liệu gốc để:
    - Lấy danh sách brand, model, year, category, engine_capacity,...
    - Tính thống kê residual theo segment cho người bán & admin.

    👉 TODO: đổi path/file cho đúng với data của bạn:
        data/motorbike_clean_for_app.csv
    """
    data_path = Path("data/motorbike_clean_for_app.csv")
    if not data_path.exists():
        st.error("❌ Không tìm thấy file data/motorbike_clean_for_app.csv\n"
                 "Hãy kiểm tra lại thư mục data và tên file.")
        st.stop()
    df = pd.read_csv(data_path)

    # Nếu có year mà chưa có years_used thì tạo thêm
    if "year" in df.columns and "years_used" not in df.columns:
        current_year = 2025  # có thể điều chỉnh
        df["years_used"] = current_year - df["year"]

    # Nếu chưa có engine_capacity thì tạo tạm để app không lỗi
    if "engine_capacity" not in df.columns:
        df["engine_capacity"] = np.nan

    return df


def compute_segment_stats(model, df):
    """
    Từ dữ liệu gốc:
    - Dự đoán giá bằng model (fair_price)
    - Tính residual = price - fair_price
    - Tạo segment = brand__model__year
    - Tính thống kê theo segment (cho người bán & admin)
    """
    df = df.copy()

    if "price" not in df.columns:
        st.error("❌ Dữ liệu không có cột 'price'. "
                 "Cần có giá thực tế để tính residual & segment stats.")
        st.stop()

    # Chọn feature columns: ở đây đơn giản là toàn bộ trừ price
    feature_cols = [c for c in df.columns if c != "price"]
    X = df[feature_cols]

    # Dự đoán
    try:
        df["predict_price"] = model.predict(X)
    except Exception as e:
        st.error(f"⚠ Lỗi khi model.predict trên dữ liệu gốc: {e}")
        st.stop()

    # Residual
    df["resid"] = df["price"] - df["predict_price"]

    # Đảm bảo có brand/model/year
    for col in ["brand", "model", "year"]:
        if col not in df.columns:
            df[col] = "unknown"

    df["segment"] = (
        df["brand"].astype(str)
        + "__"
        + df["model"].astype(str)
        + "__"
        + df["year"].astype(str)
    )

    # Tính thống kê theo segment
    seg_stats = (
        df.groupby("segment")
        .agg(
            resid_mean=("resid", "mean"),
            resid_std=("resid", "std"),
            p10=("price", lambda x: np.nanpercentile(x, 10)),
            p90=("price", lambda x: np.nanpercentile(x, 90)),
            n=("price", "count"),
        )
        .reset_index()
    )

    return df, seg_stats


# ==========================
# CÁC TRANG UI
# ==========================

def show_team_page():
    st.subheader("👥 Tên thành viên")

    members = [
        {"Họ tên": "Giảng viên hướng dẫn: Khuất Thủy Phương"},
        {"Họ tên": "Phạm Văn Hải", "Vai trò": "Xây dựng mô hình phát hiện bất thường"},
        {"Họ tên": "Nguyễn Trần Xuân Linh", "Vai trò": "Xây dựng mô hình dự báo giá"},
    ]

    st.table(pd.DataFrame(members))
    st.info("💡 Chỉnh sửa trực tiếp danh sách này trong file du_doan_gia_xe.py nếu cần cập nhật thêm.")


def show_project_summary_page():
    st.subheader("📌 Tóm tắt dự án")

    st.markdown(
        """
### Mục tiêu
- Xây dựng mô hình **dự đoán giá xe máy cũ** dựa trên dữ liệu thực tế từ thị trường.
- Triển khai ứng dụng web giúp:
  - 👤 **Người mua**: tham khảo mức giá hợp lý cho chiếc xe quan tâm.
  - 👤 **Người bán**: đánh giá mức giá đăng bán, phát hiện các tin đăng bất thường.
  - 🛠 **Quản trị viên**: duyệt / từ chối tin đăng, gửi lý do cho người đăng.

### Nguồn dữ liệu
- Dữ liệu thu thập từ các tin đăng bán xe máy cũ trên nền tảng trực tuyến.
- Đã làm sạch:
  - Loại bỏ các bản ghi thiếu giá, thiếu hãng xe, thiếu năm sản xuất,...
  - Chuẩn hóa đơn vị giá (VND), chuyển đổi format từ "tr" sang số.
  - Chuẩn hóa số km đã đi, năm sản xuất, phân khối xe,...

### Biến đầu vào tiêu biểu
- Hãng xe (**brand**)
- Dòng xe (**model** / **model_grouped**)
- Năm sản xuất (**year**) và số năm sử dụng (**years_used**)
- Số km đã đi (**mileage**)
- Phân khối (**engine_capacity**)
- Phân khúc xe (**category**), nếu có.

### Mô hình
- Sử dụng pipeline Machine Learning (ví dụ: Random Forest, Gradient Boosting, XGBoost).
- Đánh giá hiệu quả bằng RMSE, MAE, R².
- Đóng gói toàn bộ quy trình vào một pipeline duy nhất để triển khai trên Streamlit.
"""
    )


def show_model_page():
    st.subheader("🧠 Xây dựng mô hình")

    st.markdown(
        """
#### (1) Tiền xử lý dữ liệu
- Loại bỏ outlier nặng, bản ghi lỗi / thiếu thông tin quan trọng.
- Chuẩn hóa:
  - Giá: đồng bộ về đơn vị VND.
  - Năm sản xuất → số năm sử dụng: `years_used = current_year - year`.
  - Số km đã đi, phân khối.
- Gom nhóm các model hiếm vào nhóm 'other' để tránh sparsity.

#### (2) Xây dựng pipeline
- Bước encoding:
  - One-Hot Encoding (OHE) cho các biến phân loại: brand, model, category,...
- Bước scale (nếu cần):
  - Chuẩn hóa các biến số: mileage, years_used, engine_capacity.
- Bước mô hình:
  - Sử dụng thuật toán hồi quy phi tuyến (Random Forest / Gradient Boosting / XGBoost).
- Lưu pipeline hoàn chỉnh bằng `joblib`:
  - `models/motorbike_price_pipeline.pkl`

#### (3) Đánh giá mô hình
- Chia train/test (ví dụ 80/20).
- Chỉ số đánh giá:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² (Coefficient of Determination)
- So sánh với các mô hình baseline:
  - Linear Regression, Decision Tree,...
"""
    )


def show_buyer_page(model, df, seg_stats):
    st.subheader("💰 Dự đoán giá xe (cho người mua)")

    brands = sorted(df["brand"].dropna().unique().tolist()) if "brand" in df.columns else []
    models = sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else []
    years = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else []

    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Hãng xe (brand):", options=brands)
        model_name = st.selectbox(
            "Dòng xe (model):",
            options=models,
            help="Có thể gõ để lọc nhanh model."
        )
        year = st.selectbox("Năm sản xuất (year):", options=years)

    with col2:
        years_used_default = float(2025 - int(year)) if year is not None else 5.0
        years_used = st.number_input(
            "Số năm sử dụng (years_used):",
            min_value=0.0,
            max_value=30.0,
            value=years_used_default,
            step=0.5
        )
        mileage = st.number_input(
            "Số km đã đi (mileage):",
            min_value=0.0,
            value=30000.0,
            step=1000.0
        )
        engine_capacity = st.number_input(
            "Phân khối (engine_capacity, cc):",
            min_value=50.0,
            max_value=1000.0,
            value=125.0,
            step=25.0
        )

    category = None
    if "category" in df.columns:
        category_list = sorted(df["category"].dropna().unique().tolist())
        category = st.selectbox("Phân khúc xe (category):", options=category_list)

    if st.button("🔍 Dự đoán giá tham khảo", type="primary"):
        input_dict = {
            "brand": brand,
            "model": model_name,
            "year": year,
            "years_used": years_used,
            "mileage": mileage,
            "engine_capacity": engine_capacity,
        }
        if category is not None:
            input_dict["category"] = category

        input_df = pd.DataFrame([input_dict])

        try:
            y_pred = model.predict(input_df)[0]
            st.success(f"💡 Giá dự đoán tham khảo: **{y_pred:,.0f} VND**")
        except Exception as e:
            st.error(f"Không dự đoán được. Kiểm tra lại tên cột & pipeline. Lỗi: {e}")


def show_seller_page(model, df, seg_stats):
    st.subheader("📉 Định giá & phát hiện xe bất thường (cho người bán)")

    brands = sorted(df["brand"].dropna().unique().tolist()) if "brand" in df.columns else []
    models = sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else []
    years = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else []

    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Hãng xe (brand):", options=brands, key="seller_brand")
        model_name = st.selectbox(
            "Dòng xe (model):",
            options=models,
            key="seller_model",
            help="Có thể gõ để lọc nhanh model."
        )
        year = st.selectbox("Năm sản xuất (year):", options=years, key="seller_year")

    with col2:
        mileage = st.number_input(
            "Số km đã đi (mileage):",
            min_value=0.0,
            value=30000.0,
            step=1000.0,
            key="seller_mileage"
        )
        engine_capacity = st.number_input(
            "Phân khối (engine_capacity, cc):",
            min_value=50.0,
            max_value=1000.0,
            value=125.0,
            step=25.0,
            key="seller_engine"
        )
        ask_price = st.number_input(
            "Giá muốn đăng bán (VND):",
            min_value=0.0,
            value=25000000.0,
            step=500000.0,
            key="seller_price"
        )

    years_used = float(2025 - int(year)) if year is not None else 5.0

    category = None
    if "category" in df.columns:
        category_list = sorted(df["category"].dropna().unique().tolist())
        category = st.selectbox("Phân khúc xe (category):", options=category_list, key="seller_category")

    if st.button("📌 Đánh giá mức giá & phát hiện bất thường", type="primary"):
        input_dict = {
            "brand": brand,
            "model": model_name,
            "year": year,
            "years_used": years_used,
            "mileage": mileage,
            "engine_capacity": engine_capacity,
        }
        if category is not None:
            input_dict["category"] = category

        input_df = pd.DataFrame([input_dict])

        # Dự đoán giá hợp lý
        try:
            fair_price = model.predict(input_df)[0]
        except Exception as e:
            st.error(f"Lỗi khi dự đoán giá: {e}")
            return

        segment = f"{brand}__{model_name}__{year}"
        seg_row = seg_stats[seg_stats["segment"] == segment]

        st.write("---")
        st.write(f"**Segment:** `{segment}`")

        level = "normal"  # mặc định

        if seg_row.empty:
            st.warning("⚠ Chưa có đủ dữ liệu lịch sử cho segment này. So sánh dựa trên giá dự đoán.")

            resid = ask_price - fair_price
            st.write(f"- Giá dự đoán: **{fair_price:,.0f} VND**")
            st.write(f"- Giá đăng bán: **{ask_price:,.0f} VND**")
            st.write(f"- Chênh lệch (bán - dự đoán): **{resid:,.0f} VND**")

            if resid > 5_000_000:
                st.error("🚩 Giá đăng bán **cao hơn khá nhiều** so với dự đoán.")
                level = "high"
            elif resid < -5_000_000:
                st.info("✅ Giá đăng bán **thấp hơn dự đoán**, có thể là deal tốt (hoặc xe có vấn đề).")
                level = "low"
            else:
                st.success("👍 Giá đăng bán nằm gần mức dự đoán, khá hợp lý.")
        else:
            row = seg_row.iloc[0]
            resid_mean = row["resid_mean"]
            resid_std = row["resid_std"]
            p10 = row["p10"]
            p90 = row["p90"]
            n = int(row["n"])

            resid = ask_price - fair_price
            z_score = (resid - resid_mean) / resid_std if resid_std and not np.isnan(resid_std) else np.nan

            st.write(f"- Số mẫu lịch sử trong segment: **{n}**")
            st.write(f"- Giá dự đoán: **{fair_price:,.0f} VND**")
            st.write(f"- Giá đăng bán: **{ask_price:,.0f} VND**")
            st.write(f"- Chênh lệch (bán - dự đoán): **{resid:,.0f} VND**")
            st.write(f"- Khoảng giá lịch sử (p10–p90): **{p10:,.0f} – {p90:,.0f} VND**")

            msg = ""

            if ask_price < p10:
                msg += "🚩 Giá đăng **thấp hơn nhiều** so với mức thường thấy trong lịch sử.\n\n"
                level = "low"
            elif ask_price > p90:
                msg += "🚩 Giá đăng **cao hơn nhiều** so với mức thường thấy trong lịch sử.\n\n"
                level = "high"
            else:
                msg += "✅ Giá đăng nằm trong khoảng phổ biến (p10–p90).\n\n"

            if not np.isnan(z_score):
                msg += f"- Z-score của residual: **{z_score:.2f}**\n"
                if abs(z_score) > 2:
                    msg += "👉 Residual nằm ngoài ±2σ → **xe này được xem là bất thường** so với mô hình.\n"
                    level = "anomaly"

            if level == "anomaly":
                st.error(msg)
            elif level in ["low", "high"]:
                st.warning(msg)
            else:
                st.success(msg)

        # Nếu có dấu hiệu bất thường → cho phép gửi tin cho quản trị viên
        if level in ["low", "high", "anomaly"]:
            st.write("---")
            st.info("Tin này có dấu hiệu khác thường. Có thể gửi cho **quản trị viên** để duyệt.")

            if st.button("📤 Gửi tin này cho quản trị viên duyệt"):
                pending_post = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "brand": brand,
                    "model": model_name,
                    "year": int(year),
                    "mileage": float(mileage),
                    "engine_capacity": float(engine_capacity),
                    "category": category,
                    "ask_price": float(ask_price),
                    "fair_price": float(fair_price),
                    "level": level,
                    "segment": segment,
                }

                st.session_state["pending_posts"].append(pending_post)
                st.success("✅ Đã đưa tin này vào hàng chờ cho quản trị viên duyệt.")


def show_admin_page():
    st.subheader("🛠 Khu vực quản trị viên")

    pending = st.session_state.get("pending_posts", [])

    if not pending:
        st.info("Hiện không có tin nào chờ duyệt.")
        return

    st.markdown("### 📋 Danh sách tin chờ duyệt")

    df_pending = pd.DataFrame(pending)
    st.dataframe(
        df_pending[["time", "segment", "ask_price", "fair_price", "level"]],
        use_container_width=True
    )

    # Chọn 1 tin để xử lý
    idx = st.selectbox(
        "Chọn tin để xử lý:",
        options=list(range(len(pending))),
        format_func=lambda i: f"{i+1} - {pending[i]['segment']} - {pending[i]['ask_price']:,.0f} VND"
    )

    post = pending[idx]

    st.markdown("### 🔎 Chi tiết tin đăng")
    st.write(f"- Thời gian gửi: **{post['time']}**")
    st.write(f"- Segment: **{post['segment']}**")
    st.write(f"- Mức độ: **{post['level']}**")
    st.write(f"- Giá đăng bán: **{post['ask_price']:,.0f} VND**")
    st.write(f"- Giá dự đoán: **{post['fair_price']:,.0f} VND**")
    st.write(f"- Mileage: **{post['mileage']:,.0f} km**")
    st.write(f"- Engine: **{post['engine_capacity']:.0f} cc**")
    if post.get("category") is not None:
        st.write(f"- Phân khúc: **{post['category']}**")

    st.write("---")
    decision = st.radio("Quyết định của quản trị viên:", ["Duyệt tin", "Từ chối tin"])

    if decision == "Duyệt tin":
        if st.button("✅ Xác nhận duyệt tin"):
            st.success("Tin đã được duyệt. (Demo: chỉ xoá khỏi hàng chờ trong session)")
            st.session_state["pending_posts"].pop(idx)

    else:
        reason_type = st.selectbox(
            "Lý do từ chối:",
            [
                "Giá quá cao so với mặt bằng thị trường",
                "Giá quá thấp bất thường, có thể nhập sai hoặc xe có vấn đề",
                "Thông tin xe không rõ ràng / thiếu minh bạch",
                "Tự nhập lý do khác"
            ]
        )

        custom_reason = ""
        if reason_type == "Tự nhập lý do khác":
            custom_reason = st.text_area("Nhập nội dung thông báo cho người đăng:")

        if st.button("❌ Xác nhận từ chối tin"):
            if reason_type == "Tự nhập lý do khác":
                if not custom_reason.strip():
                    st.error("Vui lòng nhập nội dung lý do từ chối.")
                    return
                final_reason = custom_reason.strip()
            else:
                final_reason = reason_type

            # Nội dung giả định gửi cho người đăng tin
            msg = f"""
Kính gửi người đăng tin,

Tin đăng xe **{post['segment']}** với mức giá **{post['ask_price']:,.0f} VND** đã bị từ chối vì lý do:

> {final_reason}

Vui lòng điều chỉnh lại thông tin hoặc giá đăng bán cho phù hợp trước khi đăng lại.

Trân trọng,
Bộ phận kiểm duyệt.
"""
            st.success("Tin đã bị từ chối. Nội dung phản hồi dự kiến gửi cho người đăng:")
            st.code(msg, language="markdown")

            # Xoá khỏi hàng chờ
            st.session_state["pending_posts"].pop(idx)


# ==========================
# MAIN APP
# ==========================

def main():
    st.set_page_config(
        page_title="Dự đoán giá xe máy cũ",
        page_icon="🛵",
        layout="wide",
    )

    st.title("🛵 Ứng dụng dự đoán giá xe máy cũ")
    st.caption("Big Data & Machine Learning — Demo dự án định giá xe máy cũ")

    # Sidebar menu
    menu = st.sidebar.radio(
        "📂 Menu",
        [
            "Tên thành viên",
            "Tóm tắt dự án",
            "Xây dựng mô hình",
            "Dự đoán giá (người mua)",
            "Định giá & phát hiện xe bất thường (người bán)",
            "Quản trị viên",
        ],
    )

    # Chỉ load model & data khi cần
    if menu in [
        "Dự đoán giá (người mua)",
        "Định giá & phát hiện xe bất thường (người bán)",
    ]:
        model = load_model()
        df = load_data()
        df_with_pred, seg_stats = compute_segment_stats(model, df)

    if menu == "Tên thành viên":
        show_team_page()
    elif menu == "Tóm tắt dự án":
        show_project_summary_page()
    elif menu == "Xây dựng mô hình":
        show_model_page()
    elif menu == "Dự đoán giá (người mua)":
        show_buyer_page(model, df_with_pred, seg_stats)
    elif menu == "Định giá & phát hiện xe bất thường (người bán)":
        show_seller_page(model, df_with_pred, seg_stats)
    elif menu == "Quản trị viên":
        show_admin_page()


if __name__ == "__main__":
    main()
