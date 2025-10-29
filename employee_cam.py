# employee_cam.py
import streamlit as st
import cv2
import requests
import time

MANAGER_IP = "10.212.169.245"  # 👈 Replace with YOUR IP (manager)
MANAGER_PORT = 6060

st.set_page_config(page_title="Employee Camera Surveillance", layout="wide")
st.title("🧍 Employee Presence Surveillance")
st.markdown("Camera will detect if person is visible or not, and send status to manager.")

start_cam = st.checkbox("Start Surveillance")

if start_cam:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access camera.")
    else:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        last_status = None

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            status = "present" if len(faces) > 0 else "absent"

            # Send status only if changed
            if status != last_status:
                try:
                    requests.post(f"http://{MANAGER_IP}:{MANAGER_PORT}/update_status", json={"status": status})
                except:
                    pass
                last_status = status

            # Draw faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            time.sleep(1)
        cap.release()