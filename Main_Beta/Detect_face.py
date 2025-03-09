import cv2
import torch
import face_recognition
import pickle
from deepface import DeepFace

# โหลดโมเดล YOLO ที่เทรนแล้ว
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/Project_face/Model/yolov5s.pt')

# โหลด Face Encodings
with open('D:/Project_face/Model/face_encodings.pkl', 'rb') as f:
    encodings_data = pickle.load(f)
known_encodings = encodings_data['encodings']
known_labels = encodings_data['labels']

def detect_faces(image_path):
    image = cv2.imread(image_path)
    
    # ใช้ YOLO ตรวจจับใบหน้า
    results = yolo_model(image)
    detections = results.xyxy[0].cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:
            face_crop = image[int(y1):int(y2), int(x1):int(x2)]
            face_encoding = face_recognition.face_encodings(face_crop)

            if face_encoding:
                matches = face_recognition.compare_faces(known_encodings, face_encoding[0])
                name = "Unknown"
                if True in matches:
                    match_index = matches.index(True)
                    name = known_labels[match_index]
                
                # ใช้ DeepFace วิเคราะห์เพิ่มเติม
                analysis = DeepFace.analyze(face_crop, actions=['emotion', 'age', 'gender'], enforce_detection=False)
                emotion = analysis[0]['dominant_emotion']
                age = analysis[0]['age']
                gender = analysis[0]['gender']

                # แสดงข้อมูลบนภาพ
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, f"{name}, {age}, {gender}, {emotion}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("YOLO + Face Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ทดสอบรัน
detect_faces("../dataset/test/sample.jpg")
