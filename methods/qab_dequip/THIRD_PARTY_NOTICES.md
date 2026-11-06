# Nguồn, tác giả và trích dẫn

## QAB

Nguồn thuật toán và mã MATLAB: [SayantanDutta95/QAB-denoising](https://github.com/SayantanDutta95/QAB-denoising), commit `0ec04ac1ae9f5022867f5f004ca67e99507c6c9f`.

Copyright (c) 2020 SayantanDutta95. MIT license gốc được giữ trong `licenses/QAB-MIT.txt`.

Sayantan Dutta, Adrian Basarab, Bertrand Georgeot, Denis Kouamé. “Quantum mechanics-based signal and image representation: Application to denoising.” IEEE Open Journal of Signal Processing, vol. 2, pp. 190–206, 2021.

## De-QuIP

Nguồn thuật toán và mã MATLAB: [SayantanDutta95/De-QuIP-Denoising](https://github.com/SayantanDutta95/De-QuIP-Denoising), commit `384e24f217ea71352fd48ff715fb84b0c4b487f8`.

Mã MATLAB do Sayantan Dutta chuẩn bị. Bản clone này không có file LICENSE riêng; thông báo này không gán giấy phép MIT của QAB cho De-QuIP. Bản Python được xây để tái chạy nghiên cứu theo yêu cầu trong workspace này; quyền đối với nguồn De-QuIP vẫn thuộc tác giả. Dữ liệu ảnh upstream không được đóng gói trong wheel/sdist.

README/source đề nghị trích dẫn các bài sau khi sử dụng:

1. Sayantan Dutta, Adrian Basarab, Bertrand Georgeot, Denis Kouamé. “A Novel Image Denoising Algorithm Using Concepts of Quantum Many-Body Theory.” Signal Processing, vol. 201, 108690, 2022. DOI: 10.1016/j.sigpro.2022.108690.
2. Cùng tác giả. “Image Denoising Inspired by Quantum Many-Body physics.” ICIP 2021, pp. 1619–1623. DOI: 10.1109/ICIP42928.2021.9506794.
3. Cùng tác giả. “Despeckling Ultrasound Images Using Quantum Many-Body Physics.” IUS 2021, pp. 1–4. DOI: 10.1109/IUS52206.2021.9593778.
4. Bài QAB nêu trên.

## Bản Python

Các file Python và harness MATLAB trong dự án này là bản chuyển đổi/tổ chức lại, không phải release Python chính thức của các tác giả. Nguồn MATLAB tham chiếu tải riêng vào `external/` dưới repo nhóm; không sửa các clone đó. Cấu hình benchmark, tối ưu và phần mở rộng được ghi ở `docs/PORTING.md` và `docs/BENCHMARK.md`.

