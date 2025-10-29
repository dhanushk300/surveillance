import streamlit as st
import cv2
import os
import time
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# -------- CONFIG ----------
MODEL_PATH = "yolov8n.pt"   # change to your custom weights later
LOG_CSV = "detections.csv"
CAPTURE_DIR = "captured_images"
IMG_SAVE_QUALITY = 90
FPS_SLEEP = 0.03
# --------------------------

st.set_page_config(page_title="Smart Desk Surveillance", layout="wide")
st.title("🧠 Smart Desk Surveillance System")
st.markdown("Real-time monitoring — Object Detection & Person Surveillance")

# Sidebar navigation
mode = st.sidebar.radio("Select Mode", ["🏷 Object Detection", "🧍 Person Surveillance"])

# Ensure directories exist
os.makedirs(CAPTURE_DIR, exist_ok=True)
if not os.path.exists(LOG_CSV):
    pd.DataFrame(columns=["timestamp","object","confidence","x1","y1","x2","y2"]).to_csv(LOG_CSV, index=False)

# -------------------------------
# MODE 1: OBJECT DETECTION
# -------------------------------
if mode == "🏷 Object Detection":
    st.header("🎯 Real-Time Object Detection")
    confidence = st.slider("Detection confidence", 0.25, 1.0, 0.5)
    start_camera = st.checkbox("Start Camera")

    @st.cache_resource
    def load_model(path):
        return YOLO(path)

    model = load_model(MODEL_PATH)

    frame_placeholder = st.empty()
    chart_placeholder = st.empty()
    events_placeholder = st.empty()

    def log_detection(obj_name, conf, bbox):
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "object": obj_name,
            "confidence": round(float(conf), 3),
            "x1": int(bbox[0]), "y1": int(bbox[1]), "x2": int(bbox[2]), "y2": int(bbox[3])
        }
        pd.DataFrame([row]).to_csv(LOG_CSV, mode="a", header=False, index=False)

    if start_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("⚠ Camera not found. Check permissions.")
        else:
            st.success("✅ Camera started successfully.")

        while start_camera:
            ret, frame = cap.read()
            if not ret:
                st.warning("Frame not available.")
                break

            results = model(frame, conf=confidence)
            annotated = results[0].plot()

            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                name = model.names[cls_id]
                conf_score = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                log_detection(name, conf_score, (x1,y1,x2,y2))

            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

            df_all = pd.read_csv(LOG_CSV)
            counts = df_all['object'].value_counts().rename_axis('object').reset_index(name='count')
            if counts.shape[0] > 0:
                chart_placeholder.bar_chart(data=counts.set_index('object'))
            else:
                chart_placeholder.text("No detections yet")

            events_placeholder.dataframe(df_all.tail(10)[::-1])
            time.sleep(FPS_SLEEP)

        cap.release()
    else:
        st.info("☝ Tick 'Start Camera' to begin detection.")

# -------------------------------
# MODE 2: PERSON SURVEILLANCE
# -------------------------------
elif mode == "🧍 Person Surveillance":
    st.header("🧍 Real-Time Person Surveillance")
    st.markdown("📸 The camera feed will appear below. If a person is detected, it shows 🟢; otherwise 🔴.")

    start_surveillance = st.checkbox("Enable Surveillance Camera")
    FRAME_WINDOW = st.image([])
    status_text = st.empty()

    if start_surveillance:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Cannot access webcam.")
        else:
            st.success("✅ Surveillance Active — Stay in front of camera.")

        last_status = None
        last_time = time.time()

        while start_surveillance:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠ Unable to read camera frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if len(faces) > 0:
                status_text.markdown("<h3 style='color:green;'>🟢 Person is under camera surveillance</h3>", unsafe_allow_html=True)
                current_status = "present"
            else:
                status_text.markdown("<h3 style='color:red;'>🔴 Person has left the camera view</h3>", unsafe_allow_html=True)
                current_status = "absent"

            # Log change in status
            if current_status != last_status:
                now_str = time.strftime("%Y-%m-%d %H:%M:%S")
                st.write(f"[{now_str}] Status changed: {current_status.upper()}")
                last_status = current_status

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            time.sleep(0.05)

        cap.release()
    else:
        st.info("☝ Tick 'Enable Surveillance Camera' to start monitoring.")