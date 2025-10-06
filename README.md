Hệ thống xử lý CCCD/CMND (FastAPI + YOLO + OCR)

Tổng quan
Dự án cung cấp một pipeline hoàn chỉnh để xử lý giấy tờ tùy thân Việt Nam (CMND/CCCD): phát hiện 4 góc và cắt theo phối cảnh, nhận dạng loại giấy tờ (cũ/mới), và trích xuất văn bản/Thông tin bằng các mô hình OCR. Ứng dụng FastAPI cung cấp các REST API đơn giản cùng giao diện web (Jinja2) để tải ảnh, cắt ảnh và trích xuất thông tin.

Tính năng chính
- Phát hiện khu vực thẻ và cắt chuẩn hóa bằng YOLO
- Phân loại loại giấy tờ (CCCD cũ/CCCD mới) bằng các mô hình YOLO riêng
- Chiến lược OCR nhiều mô hình (VietOCR finetune + VietOCR base; tùy chọn PaddleOCR dự phòng)
- REST API đơn giản và giao diện web để thử nghiệm nhanh và tích hợp

Kiến trúc
- Web/API: FastAPI (kích hoạt CORS)
- Phát hiện: Ultralytics YOLO
- OCR:
  - VietOCR (finetune) cho nhận dạng chính
  - VietOCR (vgg_transformer) làm mô hình thứ hai
  - PaddleOCR làm dự phòng (tùy chọn)

Cấu trúc thư mục
- main.py                Ứng dụng FastAPI (endpoints, mount static, CORS)
- cropper.py             Phát hiện góc, cắt, và biến đổi phối cảnh
- m_ocr.py               Luồng OCR (tải mô hình, nhận dạng, phân tích)
- requirements.txt       Danh sách dependency Python
- detect_4goc/           Trọng số YOLO cho phát hiện 4 góc
- detect_ttin/           Trọng số YOLO cho phát hiện thông tin/loại thẻ
- finetune_vietocr/      Cấu hình và trọng số VietOCR đã finetune
- vgg_transformer.pth    Trọng số VietOCR base (file cục bộ)
- yolo11n.pt             Trọng số YOLO mẫu (nếu dùng)
- model.zip              Gói model sẵn (nếu được cung cấp)

Lưu ý: Mã nguồn kỳ vọng đường dẫn model cụ thể (xem phần Đường dẫn model). Hãy đảm bảo các file model của bạn đúng đường dẫn hoặc cập nhật code tương ứng.

Yêu cầu hệ thống
- Hệ điều hành: Windows 10/11, Linux hoặc macOS
- Python: khuyến nghị 3.8–3.12
- Phần cứng: Chạy CPU được; khuyến nghị GPU NVIDIA để tăng tốc

Bắt đầu nhanh (Windows PowerShell)
1) Tạo và kích hoạt virtual environment
   powershell
   python -m venv venv
   venv\Scripts\Activate.ps1

2) Cài đặt phụ thuộc
   powershell
   pip install --upgrade pip
   pip install -r requirements.txt

3) Đặt các file model (xem Đường dẫn model). Đảm bảo đúng vị trí như code.

4) Chạy API
   powershell
   python main.py

   Dịch vụ chạy tại: http://localhost:8000

Đường dẫn model (Model Files & Paths)
Mã sẽ tải model từ các vị trí sau. Hãy đảm bảo chúng tồn tại đúng đường dẫn (tạo thư mục nếu cần):

- Trong main.py (mô hình cắt góc):
  crop_model = YOLO("model/detect_4goc/4goc_all.pt")

- Trong m_ocr.py (phát hiện loại giấy tờ và OCR):
  self.old_id_model = YOLO("model/detect_ttin/cccd_cu.pt")
  self.new_id_model = YOLO("model/detect_ttin/cccd_moi.pt")
  VietOCR finetune:
  config = Cfg.load_config_from_file('model/finetune_vietocr/config.yml')
  config['weights'] = 'model/finetune_vietocr/transformerocr.pth'
  VietOCR base:
  config = Cfg.load_config_from_name('vgg_transformer')
  config['weights'] = 'model/vgg_transformer.pth'

Nếu kho hiện tại của bạn để trọng số ở thư mục cấp cao khác (ví dụ detect_4goc/, detect_ttin/, finetune_vietocr/), bạn có thể:
1) Tạo thư mục model/ và di chuyển nội dung vào đúng đường dẫn như trên; hoặc
2) Cập nhật đường dẫn trong mã nguồn cho khớp với bố cục hiện tại.

API Endpoints
Base URL: http://localhost:8000

- GET /
  Trả về giao diện web (Jinja2) để test thủ công.

- POST /upload
  Multipart form-data: file
  JSON trả về: { "message": "Upload thành công", "file_path": "temp_uploads/<uuid>.<ext>" }

- POST /crop
  JSON body: { "file_path": "temp_uploads/<uuid>.<ext>" }
  JSON trả về: {
    "status": "success",
    "message": "Cắt ảnh thành công",
    "cropped_image_path": "<tên file trong cropped_images/>"
  }

- POST /extract
  JSON body: { "file_path": "<tên file đã cắt trong cropped_images/>" }
  JSON trả về: {
    "status": "success",
    "id_type": "Cũ" | "Mới",
    "extracted_info": { ...các trường đã parse... }
  }

Ví dụ sử dụng (cURL)
1) Upload
   bash
   curl -F "file=@/path/to/your/image.jpg" http://localhost:8000/upload

2) Crop
   bash
   curl -H "Content-Type: application/json" \
        -d '{"file_path":"temp_uploads/<uuid>.<ext>"}' \
        http://localhost:8000/crop

3) Extract
   bash
   curl -H "Content-Type: application/json" \
        -d '{"file_path":"<cropped filename from previous step>"}' \
        http://localhost:8000/extract

Khắc phục sự cố (Windows)
PaddlePaddle/PaddleOCR
- Dùng phiên bản mới hơn tương thích Python 3.10–3.12 nếu phiên bản cũ không có trên PyPI.
- Lệnh tham khảo:
  powershell
  pip install paddlepaddle>=2.6.0
  pip install paddleocr>=2.7.0

VietOCR và lmdb trên Windows
- Nếu VietOCR 0.3.1 kéo lmdb==1.0.0 (khó build trên Windows), cài bản nhị phân trước:
  powershell
  pip install --only-binary=all lmdb
  pip install vietocr

PowerShell & virtual environment
- Kích hoạt: venv\Scripts\Activate.ps1
- Nếu bị chặn script, chạy (CurrentUser):
  powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Thư mục phát sinh khi chạy
- temp_uploads/         Ảnh tạm khi upload qua /upload
- cropped_images/       Ảnh đã cắt qua /crop

Bảo mật & Hiệu năng
- Không public dịch vụ nếu chưa có xác thực và giới hạn tần suất.
- Để thông lượng cao, dùng GPU cho PyTorch/YOLO và xử lý theo lô.
- Luôn kiểm tra/giới hạn kích thước loại file khi upload.

Giấy phép
Dự án cung cấp cho mục đích học tập/tích hợp. Hãy kiểm tra giấy phép của model/dataset bên thứ ba khi sử dụng.

Ghi nhận
- Ultralytics YOLO
- VietOCR
- PaddleOCR
- FastAPI / Uvicorn


