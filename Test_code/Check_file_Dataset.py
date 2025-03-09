import os

folder_path = "D:/Project_face/Data_set/Target"

# วนลูปค้นหาไฟล์ในโฟลเดอร์หลักและโฟลเดอร์ย่อย
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # รองรับหลายรูปแบบ
            img_path = os.path.join(root, filename)
            print(f"พบไฟล์: {img_path}")

