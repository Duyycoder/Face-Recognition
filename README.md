# Dự án Check-in Sự kiện bằng Nhận dạng Khuôn mặt

## Giới thiệu

Dự án này sử dụng công nghệ Trí Tuệ Nhân Tạo (AI), cụ thể là nhận dạng khuôn mặt, để giải quyết vấn đề check-in tại các sự kiện lớn. Mục tiêu chính là tối ưu hóa thời gian check-in, giảm thiểu hàng đợi và nâng cao trải nghiệm người tham dự. Hệ thống cho phép đăng ký khuôn mặt của người tham dự trước sự kiện và sau đó tự động nhận diện họ khi đến địa điểm.

## Mục lục

- [Tính năng](#tính-năng)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Yêu cầu cài đặt (Prerequisites)](#yêu-cầu-cài-đặt-prerequisites)
- [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Cách hoạt động](#cách-hoạt-động)
- [Đóng góp](#đóng-góp)
- [Giấy phép](#giấy-phép)

## Tính năng

- **Đăng ký khuôn mặt**: Người dùng có thể đăng ký khuôn mặt bằng cách cung cấp tên và tải lên hình ảnh chân dung.
- **Check-in bằng camera**: Hệ thống sử dụng webcam để nhận diện khuôn mặt và thực hiện check-in tự động.
- **Giao diện người dùng thân thiện**: Giao diện đơn giản, dễ sử dụng được xây dựng bằng Streamlit.
- **Xử lý phía backend**: API backend được xây dựng bằng FastAPI để xử lý các tác vụ nặng như phát hiện và nhúng khuôn mặt.
- **Lưu trữ và truy xuất embedding**: Lưu trữ vector đặc trưng (embedding) của khuôn mặt và so sánh để nhận diện.

## Công nghệ sử dụng

- **Ngôn ngữ lập trình**: Python
- **Framework Backend**: FastAPI
- **Framework Frontend (Giao diện người dùng)**: Streamlit
- **Thư viện AI/Machine Learning**:
    - OpenCV
    - ONNX Runtime (cho việc chạy các mô hình AI đã được tối ưu hóa)
    - Numpy
    - Pillow (PIL)
- **Mô hình AI**:
    - Mô hình phát hiện khuôn mặt (Face Detection)
    - Mô hình nhúng khuôn mặt (Face Embedding)
    - Mô hình xác định điểm mốc khuôn mặt (Face Landmark) (nếu có, dựa trên cấu trúc file)
- **Cơ sở dữ liệu/Lưu trữ**: Lưu trữ embedding trong tệp JSON (có thể nâng cấp lên giải pháp cơ sở dữ liệu chuyên dụng hơn).

![image](https://github.com/user-attachments/assets/02e0eef3-065c-4598-9b11-047ca8e34718)


## Yêu cầu cài đặt (Prerequisites)

- Python (phiên bản khuyến nghị: 3.8 trở lên)
- pip (Python package installer)
- Git

## Hướng dẫn cài đặt

1.  **Clone dự án về máy cá nhân:**
    ```bash
    git clone [https://github.com/0152neich/Face-Recognition.git](https://github.com/0152neich/Face-Recognition.git)
    cd Face-Recognition
    ```

2.  **Tạo và kích hoạt môi trường ảo (khuyến khích):**
    ```bash
    python -m venv venv
    # Trên Windows
    venv\Scripts\activate
    # Trên macOS/Linux
    source venv/bin/activate
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```
    Tệp `requirements.txt` bao gồm các thư viện như: `fastapi`, `numpy`, `onnxruntime`, `opencv-python`, `streamlit`, v.v.

4.  **Tải trọng số mô hình (Model Weights):**
    -   Tạo một thư mục có tên `weights` bên trong thư mục `src/common/`.
    -   Tải các tệp trọng số của mô hình từ [đường dẫn được cung cấp trong README gốc](https://drive.google.com/drive/folders/1yXVLcex64GkvVZMAHi-89Kz3bE5NGYMx?usp=sharing) và lưu chúng vào thư mục `src/common/weights/` vừa tạo.

5.  **Cấu hình môi trường (nếu cần):**
    -   Tạo tệp `.env` trong thư mục gốc của dự án dựa trên tệp `.env.example` (nếu có) hoặc dựa trên các biến môi trường được định nghĩa trong `src/common/settings/settings.py`. Các biến này có thể bao gồm đường dẫn đến các API endpoint (ví dụ: `host_detector`, `host_embedding`).

## Hướng dẫn sử dụng

Hệ thống bao gồm hai thành phần chính: Backend API và Giao diện người dùng.

### 1. Chạy Backend API (FastAPI)

-   Di chuyển vào thư mục `src`:
    ```bash
    cd src
    ```
-   Khởi chạy server FastAPI bằng Uvicorn:
    ```bash
    uvicorn main:app --reload --port 5000
    ```
    API sẽ chạy tại địa chỉ `http://localhost:5000`.

### 2. Chạy Giao diện Người dùng (Streamlit)

-   Mở một cửa sổ terminal mới (đảm bảo môi trường ảo đã được kích hoạt nếu bạn đang sử dụng).
-   Di chuyển vào thư mục `src` (nếu bạn chưa ở đó):
    ```bash
    cd path/to/Face-Recognition/src
    ```
-   Khởi chạy ứng dụng Streamlit:
    ```bash
    streamlit run app.py
    ```
    Giao diện sẽ được mở trên trình duyệt web, thường tại địa chỉ `http://localhost:8501`.

### Các chức năng chính trên giao diện:

#### Tab "Đăng ký khuôn mặt"

1.  **Nhập tên người dùng**: Điền tên của người bạn muốn đăng ký.
2.  **Chọn ảnh**: Tải lên một bức ảnh chân dung rõ mặt của người đó (định dạng JPG, JPEG, PNG).
3.  **Nhấn nút "Đăng ký"**: Hệ thống sẽ xử lý ảnh, trích xuất đặc trưng khuôn mặt và lưu lại. Chờ thông báo đăng ký thành công.

#### Tab "Check-in" (hoặc "Camera Check-in")

1.  **Mở Camera**: Đảm bảo webcam của bạn đã được kết nối và cho phép trình duyệt truy cập.
2.  **Đưa khuôn mặt vào webcam**: Canh chỉnh khuôn mặt của bạn vào trung tâm khung hình camera.
3.  **Nhấn nút "Nhận diện"**: Hệ thống sẽ chụp ảnh từ webcam, phát hiện khuôn mặt, trích xuất đặc trưng và so sánh với cơ sở dữ liệu đã đăng ký để tìm người phù hợp. Kết quả nhận diện (tên và điểm tương đồng) sẽ được hiển thị.

## Cách hoạt động

Hệ thống hoạt động dựa trên một quy trình nhận dạng khuôn mặt gồm các bước chính:

1.  **Thu thập ảnh**: Ảnh được lấy từ file tải lên (để đăng ký) hoặc từ webcam (để check-in).
2.  **Phát hiện khuôn mặt (Face Detection)**: Sử dụng một mô hình AI để xác định vị trí của khuôn mặt trong ảnh (trả về bounding box).
3.  **(Tùy chọn) Xác định điểm mốc khuôn mặt (Face Landmark)**: Xác định các điểm quan trọng trên khuôn mặt (như mắt, mũi, miệng) để hỗ trợ căn chỉnh.
4.  **(Tùy chọn) Căn chỉnh khuôn mặt (Face Alignment)**: Chuẩn hóa khuôn mặt (ví dụ: xoay, thay đổi kích thước) để cải thiện độ chính xác của bước tiếp theo.
5.  **Trích xuất đặc trưng/Nhúng khuôn mặt (Face Embedding)**: Một mô hình AI khác sẽ chuyển đổi thông tin khuôn mặt đã được căn chỉnh thành một vector số (gọi là embedding). Vector này đại diện cho các đặc điểm độc nhất của khuôn mặt đó.
6.  **Lưu trữ Embedding (Khi đăng ký)**: Vector embedding cùng với tên người dùng được lưu trữ (ví dụ: trong tệp `embedding.json`).
7.  **So sánh Embedding (Khi check-in)**: Vector embedding của khuôn mặt từ webcam được so sánh với tất cả các vector embedding đã lưu trữ bằng cách sử dụng một độ đo tương đồng (ví dụ: cosine similarity).
8.  **Trả về kết quả**: Nếu độ tương đồng vượt qua một ngưỡng nhất định, hệ thống sẽ trả về tên của người dùng tương ứng.

## Đóng góp

Nếu bạn muốn đóng góp cho dự án này, vui lòng làm theo các bước sau:

1.  Fork dự án này.
2.  Tạo một nhánh mới (`git checkout -b feature/ten-tinh-nang-moi`).
3.  Thực hiện các thay đổi của bạn.
4.  Commit các thay đổi (`git commit -m 'Add some feature'`).
5.  Push lên nhánh (`git push origin feature/ten-tinh-nang-moi`).
6.  Tạo một Pull Request mới.
