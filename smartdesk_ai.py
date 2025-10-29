import cv2
from ultralytics import YOLO
import pyttsx3
from playsound import playsound
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Load default YOLOv8 model (detects 80 COCO classes)
model = YOLO('yolov8n.pt')

# Create necessary folders
os.makedirs("captures", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# CSV log file
log_file = "logs/detection_log.csv"
if not os.path.exists(log_file):
    df = pd.DataFrame(columns=["Time", "Detected_Object", "Confidence"])
    df.to_csv(log_file, index=False)

# Initialize webcam
cap = cv2.VideoCapture(0)

print("🎥 SmartDesk Surveillance Started - Press 'q' to stop")

detection_count = 0
last_detect_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not found or access denied.")
        break

    results = model.predict(source=frame, conf=0.6, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # If confident detection, trigger alerts and save
            if conf > 0.6:
                detection_count += 1
                last_detect_time = time.time()

                # Voice alert
                engine.say(f"{label} detected")
                engine.runAndWait()

                # Play beep sound (optional)
                try:
                    playsound("alert.mp3", block=False)
                except:
                    pass  # if no sound file, skip

                # Save image of detection
                img_path = f"captures/{label}_{int(time.time())}.jpg"
                cv2.imwrite(img_path, frame)

                # Log to CSV
                new_entry = pd.DataFrame([[time.strftime("%Y-%m-%d %H:%M:%S"), label, conf]],
                                         columns=["Time", "Detected_Object", "Confidence"])
                new_entry.to_csv(log_file, mode='a', header=False, index=False)

    cv2.imshow("SmartDesk Surveillance", frame)

    # Auto-stop if no detections for 1 minute
    if time.time() - last_detect_time > 60:
        print("🕒 No detections for 60s — stopping.")
        break

    # Manual stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# After stop → show graph of detections
df = pd.read_csv(log_file)
if not df.empty:
    plt.figure(figsize=(8, 4))
    plt.title("Detections Confidence Over Time")
    plt.plot(range(len(df)), df["Confidence"], marker='o')
    plt.xlabel("Detection Index")
    plt.ylabel("Confidence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/detection_graph.png")
    plt.show()

print(f"✅ Log saved at {log_file}")
