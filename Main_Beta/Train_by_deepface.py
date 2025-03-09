import pickle
import os
from deepface import DeepFace

def create_face_encodings_deepface(image_folder, db_path):
    known_labels = []
    known_names = []

    # สร้างฐานข้อมูลการเทรน
    for folder_name in os.listdir(image_folder):
        folder_path = os.path.join(image_folder, folder_name)

        # ตรวจสอบว่าเป็นโฟลเดอร์ที่มีภาพหรือไม่
        if os.path.isdir(folder_path):
            known_labels.append(folder_name)
            known_names.append(folder_name)  # หรือใส่ชื่อจริงของบุคคล

            # เพิ่มภาพจากโฟลเดอร์นี้ไปยังฐานข้อมูลของ DeepFace
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # ใช้ DeepFace เพื่อค้นหาข้อมูลใบหน้า
                try:
                    # ตรวจสอบใบหน้าในภาพ
                    result = DeepFace.find(img_path=image_path, db_path=db_path, model_name="Facenet", enforce_detection=False)
                    
                    # แสดงผลว่าได้ทำการค้นหาหรือไม่
                    if result:
                        print(f"✅ พบใบหน้าในภาพ: {image_path}")
                    else:
                        print(f"❌ ไม่พบใบหน้าในภาพ: {image_path}")
                except Exception as e:
                    print(f"❌ Error with image {image_path}: {e}")
    
    # บันทึกข้อมูลที่ได้ลงในไฟล์ pickle
    data = {
        "labels": known_labels,
        "names": known_names
    }

    with open("face_encodings_deepface.pkl", "wb") as f:
        pickle.dump(data, f)
    
    print("✅ บันทึกข้อมูลใบหน้าเสร็จสิ้นลงในไฟล์ 'face_encodings_deepface.pkl'.")

# เรียกใช้ฟังก์ชัน
create_face_encodings_deepface("D:/Project_face/Data_set/Taget", "D:/Project_face/Model")