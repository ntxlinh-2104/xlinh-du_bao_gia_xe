import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# ==========================
#  CẤU HÌNH GIAO DIỆN
# ==========================
st.set_page_config(
    page_title="Dự đoán giá xe máy",
    layout="centered",
)

st.title("🏍️ Ứng dụng dự đoán giá xe máy cũ")

# ==========================
#  ĐƯỜNG DẪN FILE
# ==========================
DATA_PATH = "motorbike_cleaned.csv"       # dữ liệu để vẽ biểu đồ & gợi ý
MODEL_PATH = "motorbike_price_model.pkl"  # model sklearn dạng Pipeline

# ==========================
#  LOAD DATA
# ==========================
df = None
if os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.warning(f"Không đọc được file dữ liệu `{DATA_PATH}`.\nLỗi: {e}")
        df = None

# ==========================
#  LOAD MODEL
# ==========================
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Không load được model từ `{MODEL_PATH}`.\nLỗi: {e}")
else:
    st.warning(
        f"⚠️ Không tìm thấy file model `{MODEL_PATH}`.\n"
        "Một số chức năng dự đoán sẽ không hoạt động cho đến khi bạn đặt file model đúng chỗ."
    )

# ==========================
#  ẢNH BANNER (TUỲ CHỌN)
# ==========================
if os.path.exists("xe_may_cu.jpg"):
    st.image("xe_may_cu.jpg", use_container_width=True)

# ==========================
#  HÀM PHỤ
# ==========================
def get_unique_safe(col_name: str):
    """Lấy danh sách giá trị duy nhất của một cột trong df (nếu có)."""
    if df is not None and col_name in df.columns:
        return sorted(df[col_name].dropna().unique().tolist())
    return []

def predict_price(input_dict: dict):
    """Nhận dict thông tin xe → trả về giá dự đoán (nếu model tồn tại)."""
    if model is None:
        return None

    # Tạo DataFrame 1 dòng
    X_new = pd.DataFrame([input_dict])

    # Giả định model là Pipeline sklearn xử lý full features bên trong
    try:
        y_pred = model.predict(X_new)
        return float(y_pred[0])
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
        return None

def format_currency(x):
    try:
        return f"{x:,.0f} VND".replace(",", ".")
    except Exception:
        return x


# ==========================
#  SIDEBAR MENU
# ==========================
menu = st.sidebar.radio(
    "📂 Chọn nội dung",
    [
        "👥 Tên thành viên",
        "📘 Tóm tắt dự án",
        "🔧 Xây dựng mô hình",
        "💰 Dự đoán giá",
        "🚨 Xác định xe bất thường",
    ],
)

# ==========================
#  1. TÊN THÀNH VIÊN
# ==========================
if menu == "👥 Tên thành viên":
    st.subheader("👥 Thành viên nhóm")

    # Danh sách thành viên – chỉnh trong code
    members = [
        "Nguyễn Trần Xuân Linh",
        "Nguyễn Văn A",
        "Trần Thị B",
        "Lê Văn C",
    ]

    st.markdown("**Danh sách thành viên thực hiện dự án:**")
    for i, m in enumerate(members, start=1):
        st.markdown(f"- {i}. {m}")

# ==========================
#  2. TÓM TẮT DỰ ÁN
# ==========================
elif menu == "📘 Tóm tắt dự án":
    st.subheader("📘 Tóm tắt dự án")

    st.markdown(
        """
Dự án này xây dựng mô hình **dự đoán giá xe máy cũ** dựa trên dữ liệu thu thập từ Chợ Tốt.  
Bộ dữ liệu bao gồm các thông tin như:

- Thương hiệu (brand)
- Dòng xe (model)
- Phân khối (engine capacity)
- Số năm sử dụng (years_used)
- Số km đã đi (mileage)
- Khoảng giá rao bán (price, price_min, price_max, mid_price, ...)

Mục tiêu:

- Hỗ trợ **người mua** ước lượng mức giá hợp lý cho một chiếc xe cụ thể.
- Hỗ trợ **người bán** tham chiếu mức giá thị trường để **tránh bán quá rẻ hoặc rao quá cao**.
"""
    )

    if df is not None and "model" in df.columns:
        st.markdown("### 📊 Top 10 dòng xe phổ biến")
        top_models = (
            df["model"]
            .value_counts()
            .head(10)
            .sort_values(ascending=True)
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        top_models.plot(kind="barh", ax=ax)
        ax.set_xlabel("Số lượng tin đăng")
        ax.set_ylabel("Dòng xe (model)")
        ax.set_title("Top 10 dòng xe phổ biến trong dữ liệu")
        st.pyplot(fig)
    else:
        st.info("Không có dữ liệu hoặc thiếu cột `model` để vẽ biểu đồ.")

# ==========================
#  3. XÂY DỰNG MÔ HÌNH
# ==========================
elif menu == "🔧 Xây dựng mô hình":
    st.subheader("🔧 Quy trình xây dựng và lựa chọn mô hình")

    st.markdown(
        """
Quy trình xây dựng mô hình được triển khai trên **PySpark MLlib**:

1. **Tiền xử lý & làm sạch dữ liệu**  
   - Chuẩn hóa đơn vị giá, loại bỏ giá trị thiếu và ngoại lai nặng.  
   - Mã hóa biến phân loại bằng `StringIndexer` và `OneHotEncoder`.  
   - Chuẩn hóa các biến số như `years_used`, `mileage` bằng `StandardScaler`.  

2. **Tạo vector đặc trưng**  
   - Gộp tất cả các biến sau xử lý vào một cột `features` qua `VectorAssembler`.  

3. **Chia tập dữ liệu**  
   - 80% cho tập huấn luyện (train), 20% cho tập kiểm tra (test).  

4. **Huấn luyện & so sánh nhiều mô hình**  
   - Linear Regression  
   - Decision Tree Regressor  
   - Random Forest Regressor  
   - Gradient Boosted Trees (GBT)  
   - LinearSVR  
   - XGBoost Regressor  

5. **Đánh giá mô hình bằng RMSE và R²**  
   - **RMSE (Root Mean Squared Error):** càng thấp càng tốt.  
   - **R² (Coefficient of Determination):** càng cao càng tốt.  
"""
    )

    st.markdown("### 📈 Kết quả tóm tắt các mô hình")

    metrics_data = {
        "Mô hình": [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "XGBoost Regressor",
            "LinearSVR",
        ],
        "RMSE": [
            7_804_938.12,
            6_236_952.32,
            6_094_309.51,
            "Thấp nhất (không log cụ thể)",
            "Rất cao",
        ],
        "R²": [
            0.6800,
            0.7956,
            0.8049,
            "Cao nhất",
            "Âm",
        ],
        "Nhận xét": [
            "Trung bình, không bắt được phi tuyến",
            "Khá, nhưng dễ overfit",
            "Tốt, ổn định",
            "Tốt nhất, được chọn",
            "Loại",
        ],
    }

    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)

    st.markdown(
        """
**Nhận xét:**

- **Linear Regression**: R² = 0.68, RMSE ≈ 7.8 triệu → mô hình tuyến tính, giải thích biến động giá còn hạn chế.  
- **Decision Tree**: cải thiện rõ rệt, nhưng mô hình đơn cây dễ **overfit**.  
- **Random Forest**: R² ≈ 0.80, RMSE ≈ 6.09 triệu → ổn định và tốt hơn Decision Tree.  
- **LinearSVR**: cho R² âm và RMSE rất cao → mô hình không phù hợp, bị loại.  
- **XGBoost Regressor**: có **R² cao nhất** và **RMSE thấp nhất** trong tất cả mô hình.  
  → Đây là **mô hình tối ưu** được lựa chọn cho bài toán dự đoán giá xe máy cũ.
"""
    )

# ==========================
#  4. DỰ ĐOÁN GIÁ – BOX NGƯỜI MUA / NGƯỜI BÁN
# ==========================
elif menu == "💰 Dự đoán giá":
    st.subheader("💰 Dự đoán giá xe máy")

    if model is None:
        st.error("Chưa có model để dự đoán. Hãy kiểm tra lại file model.")
    else:
        # Gợi ý giá trị từ dữ liệu nếu có
        brands = get_unique_safe("brand")
        models = get_unique_safe("model")
        categories = get_unique_safe("category")
        capacities = get_unique_safe("engine_capacity")

        st.markdown("### 🔧 Thông tin chiếc xe")

        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox(
                "Thương hiệu (brand):",
                brands if brands else ["Honda", "Yamaha", "Suzuki", "Khác"],
            )

            model_name = st.selectbox(
                "Dòng xe (model):",
                models if models else ["Wave", "Air Blade", "Exciter", "SH", "Khác"],
                help="Có thể gõ để tìm nhanh trong danh sách.",
            )

            category = st.selectbox(
                "Phân khúc (category):",
                categories if categories else ["Xe số", "Tay ga", "Côn tay", "Khác"],
            )

        with col2:
            years_used = st.number_input(
                "Số năm sử dụng (years_used):",
                min_value=0.0,
                max_value=30.0,
                value=5.0,
                step=0.5,
            )

            mileage = st.number_input(
                "Số km đã đi (mileage):",
                min_value=0.0,
                max_value=300_000.0,
                value=30_000.0,
                step=1_000.0,
            )

            engine_capacity = st.selectbox(
                "Phân khối (engine_capacity):",
                capacities if capacities else [110, 125, 150, 155, 175, 200],
            )

        # Tạo input chung
        input_info = {
            "brand": brand,
            "model": model_name,
            "category": category,
            "years_used": years_used,
            "mileage": mileage,
            "engine_capacity": engine_capacity,
        }

        buyer_tab, seller_tab = st.tabs(["💡 Cho người mua", "💼 Cho người bán"])

        # ===== BOX CHO NGƯỜI MUA =====
        with buyer_tab:
            st.markdown(
                """
**Mục đích:**  
- Hỗ trợ người mua ước lượng **giá thị trường hợp lý** cho chiếc xe với cấu hình đã nhập.
"""
            )

            if st.button("🚀 Dự đoán giá (cho người mua)"):
                y_hat = predict_price(input_info)

                if y_hat is not None:
                    st.success(f"✅ Giá thị trường ước tính: **{format_currency(y_hat)}**")

                    st.markdown(
                        """
Gợi ý:

- Nếu giá người bán rao **thấp hơn nhiều** so với mức này → có thể là **cơ hội tốt**, nhưng cần kiểm tra kỹ chất lượng xe.  
- Nếu giá rao **cao hơn nhiều** → nên thương lượng hoặc cân nhắc xe khác.
"""
                    )
                else:
                    st.error("Không dự đoán được giá. Vui lòng kiểm tra lại model và dữ liệu đầu vào.")

        # ===== BOX CHO NGƯỜI BÁN =====
        with seller_tab:
            st.markdown(
                """
**Mục đích:**  
- Hỗ trợ người bán so sánh **giá rao dự định** với **giá thị trường dự đoán**.  
- Kiểm tra xem giá rao **có quá cao / quá thấp** so với thị trường hay không.
"""
            )

            listed_price = st.number_input(
                "Giá rao bán dự định (VND):",
                min_value=0.0,
                max_value=200_000_000.0,
                value=30_000_000.0,
                step=500_000.0,
            )

            threshold_pct = st.slider(
                "Ngưỡng chênh lệch cho là 'bất thường' (%):",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
            )

            if st.button("🔍 Kiểm tra giá rao (cho người bán)"):
                y_hat = predict_price(input_info)

                if y_hat is None:
                    st.error("Không dự đoán được giá. Vui lòng kiểm tra lại model.")
                else:
                    diff = listed_price - y_hat
                    diff_pct = diff / y_hat * 100 if y_hat != 0 else 0.0

                    st.write(f"💡 Giá thị trường (dự đoán): **{format_currency(y_hat)}**")
                    st.write(f"💵 Giá rao bán dự định: **{format_currency(listed_price)}**")
                    st.write(f"📊 Chênh lệch tuyệt đối: **{format_currency(diff)}**")
                    st.write(f"📊 Chênh lệch tương đối: **{diff_pct:.1f}%**")

                    if abs(diff_pct) <= threshold_pct:
                        st.success("✅ Giá rao bán **hợp lý**, không có dấu hiệu bất thường lớn.")
                    elif diff_pct > threshold_pct:
                        st.warning("⚠️ Giá rao đang **cao hơn đáng kể** so với giá thị trường. Có thể cần giảm bớt nếu muốn bán nhanh.")
                    else:
                        st.info("💎 Giá rao đang **thấp hơn đáng kể** so với giá thị trường. Có thể bán được rất nhanh, nhưng cũng có nguy cơ bị bán 'hớ'.")

# ==========================
#  5. XÁC ĐỊNH XE BẤT THƯỜNG (CHỈ GIẢI THÍCH)
# ==========================
elif menu == "🚨 Xác định xe bất thường":
    st.subheader("🚨 Xác định xe bất thường")

    st.info(
        """
Chức năng **kiểm tra xe rao bán bất thường (quá rẻ / quá đắt)**  
đã được tích hợp trực tiếp vào **Box “Cho người bán”** trong mục **“💰 Dự đoán giá”**.

Vui lòng chuyển sang mục **💰 Dự đoán giá** và chọn tab **“💼 Cho người bán”** để sử dụng.
"""
    )
