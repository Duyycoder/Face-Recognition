# Face-Recognition

## Hướng dẫn cài đặt

- Đầu tiên clone dự án về máy cá nhân
```
git clone https://github.com/0152neich/Face-Recognition.git
```

- Tải dependencies:
```
pip install requirements.txt
```

- Tạo 1 folder `weights` nằm trong `src/common`, sau đó tải model từ [đường dẫn](https://drive.google.com/drive/folders/1yXVLcex64GkvVZMAHi-89Kz3bE5NGYMx?usp=sharing) và lưu vào `weights`.

- Để chạy backend, cd vào thư mục src sau đó chạy
```
uvicorn main:app --reload --port 5000
```
- Để chạy giao diện:
```
streamlit run app.py
```

## Hướng dẫn sử dụng

- Ở tab đăng ký:

    - Nhập tên và upload ảnh chân dung.
    - Sau đó nhấn nút **Đăng ký** và đợi thông báo đăng ký thành công .
- Ở tab Camera Check-in:

    - Đưa khuân mặt chân dung vào webcam và nhấn **Nhận diện** để nhận diện.