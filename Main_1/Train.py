import face_recognition
import pickle
import os

known_encodings = []
known_names = []
known_labels = []  # เพิ่มตัวแปรสำหรับเก็บ labels

folder_path = "D:/Project_face/Target"

# วนลูปทุกโฟลเดอร์ย่อย
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.png')):  # รองรับหลายรูปแบบ
            name = filename.split('-')[0]  # ดึงชื่อจากไฟล์
            img_path = os.path.join(root, filename)

            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)

            if encodings:
                known_encodings.append(encodings[0])  # เก็บ encoding
                known_names.append(name)  # เก็บชื่อ
                known_labels.append(name)  # เก็บ label สำหรับแต่ละคน

# บันทึกข้อมูล
data = {"encodings": known_encodings, "names": known_names, "labels": known_labels}  # เพิ่ม labels
with open("face_encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print(f"✅ Training Complete! Encodings saved to face_encodings.pkl (รวม {len(known_encodings)} คน)")