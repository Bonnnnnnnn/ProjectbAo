from flask import Flask, render_template, Response, jsonify
import cv2
import torch
from ultralytics import YOLO
import threading

app = Flask(__name__)

# โหลดโมเดล YOLOv8
model = YOLO('yolov8n.pt')

# โหลดโมเดล Haar Cascade สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

detection_alert = False

# ฟังก์ชันประมวลผลวิดีโอ
cap = cv2.VideoCapture(0)  # ใช้กล้องเว็บแคม

def detect_humans():
    global detection_alert
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detection_alert = False
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0].item())
                if cls == 0:  # ตรวจจับเฉพาะมนุษย์
                    detection_alert = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            cv2.putText(frame, 'Face', (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# สตรีมวิดีโอไปยังหน้าเว็บ
@app.route('/video_feed')
def video_feed():
    return Response(detect_humans(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ส่งสถานะการตรวจจับ
@app.route('/alert_status')
def alert_status():
    return jsonify({'alert': detection_alert})

# แสดงหน้าเว็บหลัก
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    threading.Thread(target=detect_humans, daemon=True).start()
    app.run(debug=True, threaded=True)
