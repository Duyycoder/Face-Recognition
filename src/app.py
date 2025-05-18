import streamlit as st
import requests
import cv2
import os
import json
import time
import threading
import numpy as np
from PIL import Image
from io import BytesIO

from common.settings import Settings
from common.utils import (
    save_embedding,
    load_embeddings,
    cosine_similarity,
)

settings = Settings()

EMBEDDING_PATH = "F:\D\\NCKH_24-25\Face-Recognition\data\embedding.json"

# Ensure folders exist
if not os.path.exists(EMBEDDING_PATH):
    with open(EMBEDDING_PATH, "w") as f:
        json.dump([], f)

def get_face_detect(image):
    _, buffer = cv2.imencode('.jpg', image)
    file_bytes = BytesIO(buffer.tobytes())
    file_bytes.name = 'image.jpg'
    files = {'file': (file_bytes.name, file_bytes, 'image/jpeg')}
    response = requests.post(str(settings.host_detector), files=files)
    if response.status_code == 200:
        return {
            "bbox": response.json()['info']['bboxes'],
            "landmarks": response.json()['info']['landmarks']
        }
    return None

def get_face_embedding(image, kpps):
    payload = {
        'image': image.tolist(),
        'landmarks': np.array(kpps[0]).tolist(),
    }
    response = requests.post(str(settings.host_embedding), json=payload)
    if response.status_code != 200:
        print("Error:", response.text)
        return None
    return response.json()['info']["face_embedding"]

# Streamlit UI
st.title("🧑‍💻 Face Registration & Recognition")

# Theo dõi tab hiện tại
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Đăng ký khuôn mặt"  # Tab mặc định

tab1, tab2 = st.tabs(["📥 Đăng ký khuôn mặt", "📸 Check-in"])

# Cập nhật tab hiện tại
with tab1:
    st.session_state.active_tab = "Đăng ký khuôn mặt"

with tab2:
    st.session_state.active_tab = "Check-in"

# --- ĐĂNG KÝ ---
with tab1:
    name = st.text_input("Nhập tên người dùng")
    st.markdown("### 📤 Chọn ảnh")
    uploaded_file = st.file_uploader("Tải ảnh khuôn mặt lên", type=["jpg", "jpeg", "png"])
    upload_btn = st.button("Đăng ký")

    def handle_registration(image_np, source="webcam"):
        if name.strip() == "":
            st.warning("⚠️ Vui lòng nhập tên.")
            return

        # Gọi API detect khuôn mặt
        with st.spinner("Đang xử lý ảnh và đăng ký..."):
            face_info = get_face_detect(image=image_np)
            if face_info is None or not face_info.get("bbox"):
                st.error("❌ Không phát hiện khuôn mặt.")
                return

        # Gọi API embedding
        emb = get_face_embedding(image=image_np, kpps=face_info["landmarks"])
        if emb:
            save_embedding(name, emb, EMBEDDING_PATH)
            st.success(f"✅ Đăng ký thành công từ {source}!")
        else:
            st.error("❌ Không thể tạo embedding.")

    if upload_btn:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            handle_registration(image_np, source="upload")
        else:
            st.warning("⚠️ Vui lòng tải lên một ảnh.")

with tab2:
    st.subheader("📸 Camera Check-in")
    placeholder = st.empty()
    result_placeholder = st.empty()

    # Mở camera khi vào tab2
    if st.session_state.active_tab == "Check-in":
        if 'cap' not in st.session_state or not st.session_state.cap.isOpened():
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Tắt camera khi rời tab2
    if st.session_state.active_tab != "Check-in":
        if 'cap' in st.session_state and st.session_state.cap.isOpened():
            st.session_state.cap.release()
        st.session_state.pop('cap', None)

    # Nhấn nhận diện sẽ gán cờ để xử lý riêng 1 frame
    if 'recognize_flag' not in st.session_state:
        st.session_state.recognize_flag = False

    if st.button("Nhận diện"):
        st.session_state.recognize_flag = True

    # Cam live loop (chỉ khi ở tab Check-in)
    cap = st.session_state.get("cap", None)
    if cap and cap.isOpened() and st.session_state.active_tab == "Check-in":
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Không đọc được frame từ webcam.")
                break

            frame = cv2.flip(frame, 1)
            placeholder.image(frame, channels="BGR", use_container_width=True)

            # Nếu nhấn nút nhận diện thì xử lý frame đó
            if st.session_state.recognize_flag:
                face_info = get_face_detect(image=frame)
                if face_info and face_info.get("bbox"):
                    emb = get_face_embedding(image=frame, kpps=face_info["landmarks"])
                    if emb:
                        data = load_embeddings(embedding_path=EMBEDDING_PATH)
                        if not data:
                            result_placeholder.warning("⚠️ Chưa có ai được đăng ký.")
                        else:
                            scores = [(person["name"], cosine_similarity(emb, person["embedding"])) for person in data]
                            scores = sorted(scores, key=lambda x: x[1], reverse=True)
                            best_name, best_score = scores[0]
                            if best_score > 0.4:
                                result_placeholder.success(f"✅ **{best_name}** (score: {best_score:.2f})")
                            else:
                                result_placeholder.warning(f"⚠️ Không nhận diện được (score: {best_score:.2f})")
                    else:
                        result_placeholder.error("❌ Không thể tạo embedding.")
                else:
                    result_placeholder.error("❌ Không phát hiện khuôn mặt.")
                # Reset flag sau khi xử lý
                st.session_state.recognize_flag = False

            # Dừng 0.05s để cập nhật frame mới (loop giả)
            time.sleep(0.05)

            # Điều kiện dừng vòng lặp nếu rời tab
            if st.session_state.active_tab != "Check-in":
                break
