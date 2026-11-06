# Đối chiếu Python và MATLAB

Chạy PowerShell từ thư mục gốc repo nhóm. Các lệnh dưới đây dùng dữ liệu được tạo ở lần benchmark của bạn; số seed giống nhau không bảo đảm cùng nhiễu NumPy/MATLAB.

## Tạo dữ liệu dùng chung

```powershell
.\.venv-qab\Scripts\python.exe scripts/fetch_upstream.py qab dequip
.\.venv-qab\Scripts\python.exe -m quantum_denoising benchmark --image external/De-QuIP-Denoising/De-QuIP/data/cameraman.png --sizes 64 --repo-root external --out outputs/matlab_compare_run01
```

Output gồm clean/noisy/config, output float và timings trong `case_*.mat`. Thư mục output phải mới. Mặc định Gaussian SNR22, seed2, một warmup và ba lần đo, BLAS một luồng. QAB chọn ngưỡng bằng ảnh sạch (oracle); De-QuIP không dùng clean.

## Chạy MATLAB

Mở MATLAB tại thư mục gốc repo, rồi chạy:

```matlab
addpath(fullfile(pwd, 'methods', 'qab_dequip', 'matlab'));
benchmark_upstream(fullfile(pwd, 'external'), ...
    fullfile(pwd, 'outputs', 'matlab_compare_run01'));
```

Đối với MATLAB Online, tải các MAT cùng harness và các hàm upstream lên Online, giữ cấu trúc thư mục hoặc truyền đúng đường dẫn. Harness từ chối ghi đè file đo cũ. `matlab/package_online.py` là tiện ích đóng gói bộ ca lịch sử cụ thể (`report_dequip`, `report_dequip24`, `report_qab_ms3`, `report_qab_ms20`, `qab_signal`); nó không tự tạo các ca đó và không dùng cho một benchmark đơn lẻ.

## So đầu ra

Tải file `case_*_matlab.mat` về cạnh file Python `case_*.mat`, rồi chạy:

```powershell
.\.venv-qab\Scripts\python.exe -m quantum_denoising compare-matlab outputs/matlab_compare_run01
```

Python kiểm tra clean/noisy/config, hash và timings trước khi so output. CLI trả lỗi nếu sai số lớn nhất vượt tolerance (mặc định 1e-6). Chưa có file MATLAB thì không suy ra kết quả MATLAB.

## Quy ước và giới hạn

- Đo toàn bộ hàm denoise, không tính import, đọc ảnh, tạo nhiễu, metric, vẽ hình. Giữ số luồng, warmup và repeat; không chạy các benchmark đồng thời.
- PSNR mặc định peak255; SSIM Gaussian 3×3, sigma1.5, valid/population. Không clip trước khi tính metric. QAB mẫu có thang riêng; JSON lưu quy ước gốc khi khác.
- Tối ưu Python và phần cứng là các yếu tố khác biệt. MATLAB Online và máy local khác CPU, vì thế tỷ lệ thời gian không chứng minh ưu thế riêng của ngôn ngữ.
- 36 kiểm thử Python và các số liệu lịch sử không thay thế đối chiếu MATLAB runtime trên cùng MAT.
- Dữ liệu thô MAT và checkpoint không được commit. `results/historical` chỉ giữ các báo cáo nhỏ đã chọn.

