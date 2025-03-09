import os
import cv2

def rename_and_resize_images(source_folder, target_folder, rename_map, img_size=(640, 780)):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for person in os.listdir(source_folder):
        person_folder = os.path.join(source_folder, person)
        
        if not os.path.isdir(person_folder):
            continue
        
        # ตรวจสอบว่ามีการเปลี่ยนชื่อโฟลเดอร์หรือไม่
        new_person_name = rename_map.get(person, person)
        target_person_folder = os.path.join(target_folder, new_person_name)
        
        if not os.path.exists(target_person_folder):
            os.makedirs(target_person_folder)
        
        files = [f for f in os.listdir(person_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        files.sort()
        print(f"📂 Processing folder: {person} -> {new_person_name} ({len(files)} images)")

        for idx, file in enumerate(files, start=1):
            img_path = os.path.join(person_folder, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img_resized = cv2.resize(img, img_size)
            new_filename = f"{new_person_name}{idx:02d}.png"
            
            cv2.imwrite(os.path.join(target_person_folder, new_filename), img_resized)

        print(f"✅ Processed {person} -> {new_person_name}: {len(files)} images resized & renamed.")

# รับค่าเปลี่ยนชื่อโฟลเดอร์จากผู้ใช้
source_folder = 'D:/Project_face/Data_set/Raw_data'
target_folder = 'D:/Project_face/Data_set/Target'
rename_map = {}

for person in os.listdir(source_folder):
    person_folder = os.path.join(source_folder, person)
    if os.path.isdir(person_folder):
        new_name = input(f"Enter new name for '{person}' (or press Enter to keep the same): ").strip()
        if new_name:
            rename_map[person] = new_name

rename_and_resize_images(source_folder, target_folder, rename_map)

