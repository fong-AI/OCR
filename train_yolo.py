import os
from ultralytics import YOLO
import cv2

model = YOLO("model/detect_ttin/cccd_moi.pt")
# folder_path = "dataset/cmnd/images_augmented"
# image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png", ".jpeg"))]

# output_dir = "runs/detect/predict36/labels"

# for img_file in image_files:
#     img_path = os.path.join(folder_path, img_file)

#     # Detect và lưu txt tạm thời
#     results = model(img_path, save=True, save_txt=True)

#     # Tên file txt YOLO đã tạo
#     txt_name = os.path.splitext(img_file)[0] + ".txt"
#     txt_path = os.path.join(output_dir, txt_name)

#     # Nếu file txt tồn tại → sửa lại thứ tự class_id
#     if os.path.exists(txt_path):
#         with open(txt_path, "r") as f:
#             lines = f.readlines()

#         # Parse lại các dòng
#         parsed = []
#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) >= 5:
#                 cls_id = int(parts[0])
#                 parsed.append((cls_id, line.strip()))

#         # Sắp xếp theo class_id
#         parsed.sort(key=lambda x: x[0])

#         # Ghi đè lại file theo thứ tự mới
#         with open(txt_path, "w") as f:
#             for _, line in parsed:
#                 f.write(line + "\n")

#         print(f"Đã sửa thứ tự label trong: {txt_name}")
#     else:
#         print(f"Không tìm thấy label cho: {img_file}")
if __name__ == "__main__":
    # folder_path = "dataset/images_roboflow2"
    # image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png", ".jpeg"))]

    # for img_file in image_files:
    #     img_path = os.path.join(folder_path, img_file)
    #     results = model(img_path, save=True, save_txt=True)

    #     print(f"Kết quả cho {img_file}:")
    #     for r in results:
    #         print(r)
    #result = model.predict(source="cropped_images/cropped_eb564b4a08564a5787b5072c04a3153e.jpg", save=True)
    Train = model.train(data="cccd.yaml", epochs=100, imgsz=640, batch=2, patience=30)