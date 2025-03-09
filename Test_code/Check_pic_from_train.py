import pickle

# โหลดไฟล์ face_encodings.pkl
with open("face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

# แสดงจำนวนใบหน้าที่บันทึก
print(f"จำนวนใบหน้าที่บันทึก: {len(data['encodings'])}")

# แสดงชื่อบุคคลทั้งหมด
print("รายชื่อบุคคลที่บันทึก:")
print(data["names"])