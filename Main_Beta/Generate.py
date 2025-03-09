import cv2
import torch
import os

# โหลดโมเดล YOLO เพื่อตรวจจับใบหน้า
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# ตั้งค่าพาธสำหรับบันทึกภาพใบหน้า
output_folder = "D:/Project_face/Data_set/Train/"
os.makedirs(output_folder, exist_ok=True)

# โหลดภาพจากโฟลเดอร์
input_folder = "D:/Project_face/Data_set/Unlabel/"
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    
    # ใช้ YOLO ตรวจจับใบหน้า
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:
            face_crop = image[int(y1):int(y2), int(x1):int(x2)]
            face_filename = f"{output_folder}/face_{image_file.split('.')[0]}_{i}.jpg"
            cv2.imwrite(face_filename, face_crop)
            print(f"Saved: {face_filename}")
