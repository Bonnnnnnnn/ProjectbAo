import cv2
import face_recognition
import pickle
import numpy as np

# ฟังก์ชันโหลดข้อมูล pickle (ทั้ง encodings และ labels)
def load_face_model(model_path='D:/Project_face/Model/face_encodings.pkl'):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['encodings'], model_data['labels']  # คืนทั้ง encodings และ labels

# ฟังก์ชันเปรียบเทียบใบหน้า
def compare_faces(known_encodings, known_labels, face_encoding):
    # เปรียบเทียบใบหน้าทุกใบหน้าที่บันทึกไว้ใน pickle
    results = face_recognition.compare_faces(known_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)  # คำนวณความต่าง

    best_match_index = np.argmin(face_distances)  # หาค่าความแตกต่างที่น้อยที่สุด
    match_score = 100 - (face_distances[best_match_index] * 100)  # คำนวณเปอร์เซ็นต์ความแม่นยำ

    if results[best_match_index]:  # ถ้ามีการจับคู่
        person_name = known_labels[best_match_index]
        # ตัดชื่อออกจากนามสกุลหรือเลขภายหลังชื่อ
        person_name = person_name.split('.')[0]  # ตัด ".png" หรือ ".jpg" ออก
        person_name = ''.join([i for i in person_name if not i.isdigit()])  # ตัดตัวเลขออก
        return person_name, match_score

    return "Unknown", 0.0  # ถ้าไม่พบใบหน้าที่ตรง

# เริ่มต้นโปรแกรม
def start_face_recognition_system():
    # โหลดโมเดลใบหน้าจาก pickle
    known_encodings, known_labels = load_face_model('face_encodings.pkl')

    # ใช้เว็บแคม (0 = เว็บแคมหลัก)
    cap = cv2.VideoCapture(0)

    # ตรวจสอบว่าเปิดเว็บแคมสำเร็จหรือไม่
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam")
            break

        # ตรวจจับใบหน้าจากเว็บแคม
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # ตรวจสอบว่ามีใบหน้าถูกตรวจจับหรือไม่
        print(f"Detected faces: {len(face_locations)}")

        for face_encoding in face_encodings:
            person_name, match_score = compare_faces(known_encodings, known_labels, face_encoding)

            # แสดงผลการตรวจจับ
            if person_name == "Unknown":
                print("Detected: Unknown")
            else:
                print(f"Detected: {person_name} (Confidence: {match_score:.2f}%)")

            # วาดกรอบบนใบหน้า
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # แสดงชื่อบุคคลและคะแนนความแม่นยำบนกรอบใบหน้า
                cv2.putText(frame, f"{person_name} ({match_score:.2f}%)", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # แสดงผลหน้าต่าง
        cv2.imshow("Face Recognition", frame)

        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_face_recognition_system()  # เรียกใช้งานเว็บแคม
