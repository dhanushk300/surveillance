# app.py
import streamlit as st
import cv2
import os
import time
import tempfile
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# -------- CONFIG ----------
MODEL_PATH = "yolov8n.pt"   # change to your custom weights later
LOG_CSV = "detections.csv"
CAPTURE_DIR = "captured_images"
IMG_SAVE_QUALITY = 90
FPS_SLEEP = 0.03  # loop delay
# --------------------------

st.set_page_config(page_title="Smart Desk Surveillance (Advanced)", layout="wide")

st.title("🧠 Smart Desk Surveillance — Advanced")
st.markdown("Real-time detection + live analytics + alerts + screenshot capture")

# Sidebar controls (single creation; unique keys)
st.sidebar.header("Camera & Detection")
confidence = st.sidebar.slider("Min confidence", 0.25, 1.0, 0.5, key="conf")
start_camera = st.sidebar.checkbox("Start Camera", value=False, key="start_cam")
save_screenshots = st.sidebar.checkbox("Save screenshots on alert", value=True, key="save_ss")
alert_objects = st.sidebar.multiselect("Alert when these objects appear", options=[
    "person", "cell phone", "book", "bottle", "laptop"], default=["cell phone"], key="alerts")
snapshot_seconds = st.sidebar.number_input("Min seconds between snapshots (per object)", min_value=1, max_value=600, value=10, key="snapsec")

# Ensure folders and files exist
os.makedirs(CAPTURE_DIR, exist_ok=True)
if not os.path.exists(LOG_CSV):
    pd.DataFrame(columns=["timestamp","object","confidence","x1","y1","x2","y2"]).to_csv(LOG_CSV, index=False)

# Load model
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_PATH)

# placeholders for UI
left_col, right_col = st.columns((2, 1))
frame_placeholder = left_col.empty()
chart_placeholder = right_col.empty()
events_placeholder = right_col.empty()
summary_placeholder = st.container()

# helper: append detection to CSV
def log_detection(obj_name, conf, bbox):
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "object": obj_name,
        "confidence": round(float(conf), 3),
        "x1": int(bbox[0]), "y1": int(bbox[1]), "x2": int(bbox[2]), "y2": int(bbox[3])
    }
    df = pd.DataFrame([row])
    df.to_csv(LOG_CSV, mode="a", header=False, index=False)

# helper: read recent events
def read_events(n=20):
    try:
        df = pd.read_csv(LOG_CSV)
        return df.tail(n)
    except:
        return pd.DataFrame(columns=["timestamp","object","confidence"])

# helper: save captured frame for object
last_snapshot_time = {}  # in-memory throttle per object

def maybe_save_snapshot(obj_name, frame):
    now = time.time()
    last = last_snapshot_time.get(obj_name, 0)
    if now - last < snapshot_seconds:
        return None
    # save
    fname = f"{obj_name}_{int(now)}.jpg"
    path = os.path.join(CAPTURE_DIR, fname)
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path, quality=IMG_SAVE_QUALITY)
    last_snapshot_time[obj_name] = now
    return path

# run camera loop when checkbox ON
if start_camera:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open camera. Check permissions.")
    else:
        try:
            while st.session_state.start_cam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Camera frame not available.")
                    break

                # Detect
                results = model(frame, conf=confidence)
                annotated = results[0].plot()

                # Write detection logs and handle alerts
                detected_names = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    name = model.names[cls_id]
                    conf_score = float(box.conf[0].item())
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())

                    detected_names.append(name)
                    # log every detection row (you can change to only log once per ID)
                    log_detection(name, conf_score, (x1,y1,x2,y2))

                    # Alert + screenshot
                    if name in alert_objects:
                        saved = None
                        if save_screenshots:
                            saved = maybe_save_snapshot(name, frame)
                        # show banner (visual alert)
                        st.experimental_set_query_params()  # trigger rerun safety
                        st.sidebar.warning(f"ALERT: {name} detected (conf={conf_score:.2f})")
                        if saved:
                            st.sidebar.info(f"Saved snapshot: {saved}")

                # update frame
                frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

                # live chart: counts of each object (last 500 rows)
                df_all = pd.read_csv(LOG_CSV)
                recent = df_all.tail(500)
                counts = recent['object'].value_counts().rename_axis('object').reset_index(name='count')
                if counts.shape[0] > 0:
                    chart_placeholder.bar_chart(data=counts.set_index('object'))
                else:
                    chart_placeholder.text("No detections yet")

                # recent events table
                events_df = read_events(10)
                events_placeholder.subheader("Recent Events")
                events_placeholder.dataframe(events_df[::-1])

                # summary
                with summary_placeholder:
                    st.markdown("---")
                    st.subheader("Summary")
                    total = len(df_all)
                    top = recent['object'].mode()[0] if not recent['object'].empty else "N/A"
                    st.write(f"Total detections logged: **{total}**")
                    st.write(f"Most frequent object (recent): **{top}**")
                    st.write(f"Saved snapshots folder: `{CAPTURE_DIR}`")

                # small sleep to avoid hogging CPU
                time.sleep(FPS_SLEEP)

        except Exception as e:
            st.error(f"Camera loop error: {e}")
        finally:
            cap.release()
else:
    frame_placeholder.info("Click **Start Camera** in the sidebar to begin detection.")

# small note + instructions
st.markdown("### Notes")
st.markdown("- The app logs every detection to `detections.csv` in the project folder.")
st.markdown("- Use the sidebar to configure objects to alert on and whether to save screenshots.")
