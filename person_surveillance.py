# person_surveillance.py
import cv2
import socket
import os
import time

# 🔹 Replace with Manager's IP (your laptop IP)
MANAGER_IP = "10.212.169.245"
MANAGER_PORT = 6060

def send_image_to_manager(image_path):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((MANAGER_IP, MANAGER_PORT))

        filename = os.path.basename(image_path)
        filesize = os.path.getsize(image_path)

        # Send filename length + filename + filesize + filedata
        s.send(len(filename).to_bytes(4, "big"))
        s.send(filename.encode())
        s.send(filesize.to_bytes(8, "big"))

        with open(image_path, "rb") as f:
            s.sendall(f.read())

        s.close()
        print(f"📤 Sent {filename} to manager successfully.")
    except Exception as e:
        print("❌ Error sending image:", e)

def capture_and_send(frame):
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"person_{ts}.jpg"
    cv2.imwrite(fname, frame)
    send_image_to_manager(fname)
    os.remove(fname)  # optional cleanup

def main():
    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    last_capture_time = 0
    capture_interval = 5  # seconds between captures

    print("🧍 Person Surveillance started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            now = time.time()
            if now - last_capture_time >= capture_interval:
                print("👀 Person detected — capturing & sending...")
                capture_and_send(frame)
                last_capture_time = now

        # optional live preview
        cv2.imshow("Developer Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()