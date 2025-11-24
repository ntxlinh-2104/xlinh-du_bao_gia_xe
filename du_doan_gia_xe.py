import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import re
import os
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================
#  CẤU HÌNH GIAO DIỆN
# ==========================
st.set_page_config(page_title="Dự đoán giá xe máy", layout="wide")

# ==========================
#  KHỞI TẠO HÀNG CHỜ CHO QUẢN TRỊ VIÊN
# ==========================
if "pending_posts" not in st.session_state:
    st.session_state["pending_posts"] = []

# ==========================
#  HÀM LOAD DATA
# ==========================
DATA_PATH = "motorbike_cleaned.csv"


@st.cache_data
def load_data():
    df_local = None
    if os.path.exists(DATA_PATH):
        try:
            df_local = pd.read_csv(DATA_PATH)
        except Exception:
            df_local = None
    return df_local


df = load_data()

# ==========================
#  ẢNH BANNER & BIỂU ĐỒ TOP 5 (dùng ở trang Tóm tắt)
# ==========================
def show_banner_and_top5():
    # Ảnh banner
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
#  LOAD MODEL (DÙNG JOBLIB, FALLBACK PICKLE)
# ==========================
@st.cache_resource
def load_model():
    """
    Thử lần lượt:
    - motobike_price_model.joblib (joblib)
    - motobike_price_model.pkl (pickle)
    """
    candidates = [
        ("motobike_price_model.joblib", "joblib"),
        ("motobike_price_model.pkl", "pickle"),
    ]
    for path, kind in candidates:
        if os.path.exists(path):
            try:
                if kind == "joblib":
                    model_local = joblib.load(path)
                else:
                    with open(path, "rb") as f:
                        model_local = pickle.load(f)
                return model_local
            except Exception as e:
                st.error(f"Lỗi khi load model từ {path}: {e}")
                st.stop()

    st.error(
        "❌ Không tìm thấy file mô hình: motobike_price_model.joblib hoặc motobike_price_model.pkl.\n"
        "Hãy kiểm tra lại thư mục và tên file model."
    )
    st.stop()


# Lúc train model dùng các feature này:
expected_features = ["mileage", "years_used", "model", "category"]
numeric_features = ["mileage", "years_used"]
categorical_features = ["model", "category"]

# ==========================
#  DROPDOWN OPTIONS TỪ DATA (DÙNG CHUNG)
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
#  CÁC TRANG TRONG MENU
# ==========================

def page_team():
    st.subheader("👥 Tên thành viên")

    members = [
        {"Họ tên": "Giảng viên hướng dẫn: Khuất Thủy Phương"},
        {"Họ tên": "Phạm Văn Hải", "Vai trò": "Xây dựng mô hình phát hiện bất thường"},
        {"Họ tên": "Nguyễn Trần Xuân Linh", "Vai trò": "Xây dựng mô hình dự báo giá"},
    ]
    st.table(pd.DataFrame(members))
    st.info("💡 Có thể chỉnh sửa danh sách này trực tiếp trong file du_doan_gia_xe.py.")


def page_summary():
    st.subheader("📌 Tóm tắt dự án")
    show_banner_and_top5()

    st.markdown(
        """
### Mục tiêu
- Xây dựng mô hình **dự đoán giá xe máy cũ** dựa trên dữ liệu thực tế.
- Triển khai ứng dụng hỗ trợ:
  - 👤 **Người mua**: tham khảo mức giá hợp lý.
  - 👤 **Người bán**: kiểm tra mức giá dự định đăng.
  - 🛠 **Quản trị viên**: duyệt/từ chối các tin đăng bất thường.

### Dữ liệu
- File dữ liệu sử dụng trong ứng dụng: `motorbike_cleaned.csv`.
- Các biến chính:
  - `mileage` – số km đã đi.
  - `years_used` – số năm sử dụng.
  - `model` – dòng xe.
  - `category` – loại xe.

### Mô hình
- Mô hình Machine Learning được train sẵn (lưu dưới dạng `motobike_price_model.joblib` / `.pkl`).
- Đầu vào: `mileage`, `years_used`, `model`, `category`.
- Đầu ra: **giá dự đoán** (VND).
"""
    )


def page_model():
    st.subheader("🧠 Xây dựng mô hình")

    st.markdown(
        """
#### (1) Tiền xử lý dữ liệu
- Làm sạch:
  - Loại bỏ các bản ghi thiếu thông tin quan trọng (giá, model, mileage,...).
  - Chuẩn hóa định dạng số cho cột `mileage`, `price`,...
- Tạo biến:
  - `years_used = năm hiện tại - year_sx`.
  - Chuẩn hóa `model`, `category`.

#### (2) Mô hình
- Chọn tập biến đầu vào: `mileage`, `years_used`, `model`, `category`.
- Mã hóa biến phân loại bằng kỹ thuật phù hợp (ví dụ: One-Hot Encoding).
- Huấn luyện mô hình hồi quy (sklearn) để dự báo giá.
- Lưu mô hình bằng:
  - `joblib.dump(model, "motobike_price_model.joblib")`.

#### (3) Tích hợp vào Streamlit
- Ứng dụng đọc model qua `joblib.load`.
- Người dùng nhập thông tin → tạo DataFrame với đúng tên cột → `model.predict(...)`.
- Kết quả được hiển thị dưới dạng **metric** và kèm theo đánh giá mức độ hợp lý.
"""
    )


def page_buyer():
    st.markdown("## 🚀 Dự đoán giá xe máy – Người mua")
    st.subheader("📘 Nhập thông tin xe để dự đoán")

    model = load_model()

    with st.form("form_du_doan"):
        # --- Numeric: mileage, years_used, engine_capacity ---
        c1, c2, c3 = st.columns(3)
        mileage = c1.text_input("Số km đã đi:", "15000")
        years_used = c2.text_input("Số năm sử dụng:", "2")
        engine_capacity = c3.text_input("Phân khối (cc):", "125")  # chưa đưa vào model

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
            [
                {
                    "mileage": to_number_from_str(mileage),
                    "years_used": to_number_from_str(years_used),
                    "model": model_input,
                    "category": category_input,
                }
            ]
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


def page_seller():
    st.markdown("## 🧭 Phát hiện giá đăng bán bất thường – Người bán")
    st.subheader("📦 Kiểm tra mức giá bạn định đăng")

    model = load_model()

    with st.form("form_phat_hien"):
        # --- Numeric ---
        c1s, c2s, c3s = st.columns(3)
        mileage_s = c1s.text_input("Số km đã đi:", "15000", key="seller_mileage")
        years_used_s = c2s.text_input("Số năm sử dụng:", "2", key="seller_years")
        engine_capacity_s = c3s.text_input(
            "Phân khối (cc):", "125", key="seller_cc"
        )  # chỉ hiển thị

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

        price_s = st.text_input(
            "Giá bạn muốn đăng (VND):", "20000000", key="seller_price"
        )

        submit_sell = st.form_submit_button("🧮 Kiểm tra giá có hợp lý không")

    if submit_sell:
        X_sell = pd.DataFrame(
            [
                {
                    "mileage": to_number_from_str(mileage_s),
                    "years_used": to_number_from_str(years_used_s),
                    "model": model_input_s,
                    "category": category_input_s,
                }
            ]
        ).reindex(columns=expected_features)

        seller_price = to_number_from_str(price_s)

        st.write("### Dữ liệu gửi vào mô hình (người bán)")
        st.dataframe(X_sell)

        if np.isnan(seller_price):
            st.error("Vui lòng nhập 'Giá bạn muốn đăng' là số hợp lệ.")
            return

        try:
            fair_price = float(model.predict(X_sell)[0])

            st.write("### Kết quả đánh giá giá đăng bán")
            st.write(f"- Giá hợp lý theo mô hình: **{format_vnd(fair_price)}**")
            st.write(f"- Giá bạn muốn đăng: **{format_vnd(seller_price)}**")

            if fair_price <= 0:
                st.warning(
                    "Giá dự đoán không hợp lệ (<=0). Kiểm tra lại dữ liệu đầu vào hoặc mô hình."
                )
                return

            ratio = seller_price / fair_price
            low_ok = 0.9 * fair_price
            high_ok = 1.1 * fair_price

            level = "normal"  # mức độ bất thường

            if ratio < 0.7:
                st.error(
                    "🚨 Giá **quá rẻ** so với mặt bằng dự đoán → có thể là tin bất thường hoặc bạn đang bán lỗ rất mạnh."
                )
                level = "too_low"
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
                level = "too_high"

            st.write(
                f"👉 Khoảng giá tham khảo nên đăng: **{format_vnd(low_ok)} – {format_vnd(high_ok)}**"
            )

            # ====== GỬI CHO QUẢN TRỊ VIÊN NẾU BẤT THƯỜNG MẠNH ======
            if level in ["too_low", "too_high"]:
                st.write("---")
                st.info(
                    "Tin này có dấu hiệu **bất thường mạnh** về giá. "
                    "Bạn có thể gửi cho **quản trị viên** để xem xét duyệt/từ chối."
                )

                if st.button("📤 Gửi tin này cho quản trị viên duyệt"):
                    pending_post = {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "mileage": float(to_number_from_str(mileage_s)),
                        "years_used": float(to_number_from_str(years_used_s)),
                        "model": str(model_input_s),
                        "category": str(category_input_s),
                        "ask_price": float(seller_price),
                        "fair_price": float(fair_price),
                        "level": level,
                    }
                    st.session_state["pending_posts"].append(pending_post)
                    st.success(
                        "✅ Đã đưa tin này vào hàng chờ cho quản trị viên duyệt (xem ở mục 'Quản trị viên')."
                    )

        except Exception as e:
            st.error("Lỗi khi đánh giá giá đăng bán.")
            st.exception(e)


def page_admin():
    st.subheader("🛠 Khu vực quản trị viên")

    pending = st.session_state.get("pending_posts", [])

    if not pending:
        st.info("Hiện không có tin nào chờ duyệt.")
        return

    st.markdown("### 📋 Danh sách tin chờ duyệt")

    df_pending = pd.DataFrame(pending)
    st.dataframe(
        df_pending[
            ["time", "model", "category", "ask_price", "fair_price", "level"]
        ],
        use_container_width=True,
    )

    idx = st.selectbox(
        "Chọn tin để xử lý:",
        options=list(range(len(pending))),
        format_func=lambda i: f"{i+1} - {pending[i]['model']} - {format_vnd(pending[i]['ask_price'])}",
    )

    post = pending[idx]

    st.markdown("### 🔎 Chi tiết tin đăng")
    st.write(f"- Thời gian gửi: **{post['time']}**")
    st.write(f"- Model: **{post['model']}**")
    st.write(f"- Category: **{post['category']}**")
    st.write(f"- Mức độ: **{post['level']}**")
    st.write(f"- Giá đăng bán: **{format_vnd(post['ask_price'])}**")
    st.write(f"- Giá dự đoán: **{format_vnd(post['fair_price'])}**")
    st.write(f"- Số km: **{post['mileage']:.0f} km**, Số năm sử dụng: **{post['years_used']:.1f} năm**")

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
                "Tự nhập lý do khác",
            ],
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

            msg = f"""
Kính gửi người đăng tin,

Tin đăng xe **{post['model']} ({post['category']})** với mức giá **{format_vnd(post['ask_price'])}** đã bị từ chối vì lý do:

> {final_reason}

Vui lòng điều chỉnh lại thông tin hoặc giá đăng bán cho phù hợp trước khi đăng lại.

Trân trọng,
Bộ phận kiểm duyệt.
"""
            st.success("Tin đã bị từ chối. Nội dung phản hồi dự kiến gửi cho người đăng:")
            st.code(msg, language="markdown")

            st.session_state["pending_posts"].pop(idx)


# ==========================
#  MAIN
# ==========================
def main():
    st.title("🛵 Ứng dụng dự đoán giá xe máy cũ")
    st.caption("Big Data & Machine Learning — Demo dự án định giá xe máy cũ")

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

    if menu == "Tên thành viên":
        page_team()
    elif menu == "Tóm tắt dự án":
        page_summary()
    elif menu == "Xây dựng mô hình":
        page_model()
    elif menu == "Dự đoán giá (người mua)":
        page_buyer()
    elif menu == "Định giá & phát hiện xe bất thường (người bán)":
        page_seller()
    elif menu == "Quản trị viên":
        page_admin()


if __name__ == "__main__":
    main()
