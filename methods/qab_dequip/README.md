# QAB và De-QuIP — bản Python

Bản NumPy/SciPy chạy CPU, Python >=3.10, không cần MATLAB khi khử nhiễu. Có QAB cho ảnh và tín hiệu 1D, De-QuIP cho ảnh grayscale; API, CLI, benchmark, kiểm thử số học, và bộ đối chiếu MATLAB dùng chung dữ liệu.

## Cài và chạy trên Windows

```powershell
# Chạy từ thư mục gốc image-restoration-study
py -3.10 -m venv .venv-qab
.\.venv-qab\Scripts\python.exe -m pip install -e "./methods/qab_dequip[dev]"
.\.venv-qab\Scripts\python.exe -m pytest -q methods/qab_dequip/tests
.\.venv-qab\Scripts\python.exe -m quantum_denoising --help
```

Tải dữ liệu mẫu/nguồn đã ghim commit bằng `python scripts/fetch_upstream.py qab dequip` trước. Benchmark cả hai thuật toán trên cùng ảnh/nhiễu, giữ tham số lõi của tác giả:

```powershell
.\.venv-qab\Scripts\python.exe -m quantum_denoising benchmark --image external\De-QuIP-Denoising\De-QuIP\data\cameraman.png --sizes 64 128 --repo-root external --out outputs\python_denoising_new --threads 1 --repeats 3 --warmups 1
```

Thư mục output phải mới để tránh ghi đè số đo. Gồm JSON/CSV/Markdown, ảnh so sánh, output float `.npy`, và `.mat` chứa clean/noisy/config/output/timings dùng chung với MATLAB. Benchmark mặc định dùng Gaussian SNR=22 dB, seed=2; noise được tạo một lần rồi chia sẻ nguyên mảng, không giả định RNG NumPy giống MATLAB.

Để dùng crop chính giữa giống các vùng ảnh trong báo cáo MATLAB Online của bạn:

```powershell
.\.venv-qab\Scripts\python.exe -m quantum_denoising benchmark --image external\De-QuIP-Denoising\De-QuIP\data\cameraman.png --sizes 64 128 256 --algorithms dequip --preprocess center-crop --seed 42 --repo-root external --out outputs\dequip_center_new
```

Số seed=42 vẫn không tạo cùng mảng nhiễu ở hai ngôn ngữ; muốn kiểm tra sai số từng pixel phải chia sẻ MAT. Với ảnh nhỏ như 24×24, cửa sổ tương tác vượt lưới patch và dùng chỉ số tuần hoàn giống source MATLAB.

QAB tín hiệu đúng dữ liệu và tham số demo:

```powershell
.\.venv-qab\Scripts\python.exe -m quantum_denoising benchmark-signal --signal external\QAB-denoising\Signal\sample_signal.mat --repo-root external --out outputs\qab_signal_new
```

QAB ảnh tổng hợp đúng dữ liệu demo (peak=13 là quy ước metric cố định tự chọn, PSNR gốc cũng lưu riêng):

```powershell
.\.venv-qab\Scripts\python.exe -m quantum_denoising benchmark --image external\QAB-denoising\Image\sample_image.mat --sizes 64 --algorithms qab --snr 15 --noise heteroscedastic --peak 13 --repo-root external --out outputs\qab_sample_new
```

Khử nhiễu ảnh của bạn (đầu vào đã có nhiễu, lệnh này không tự thêm nhiễu):

```powershell
.\.venv-qab\Scripts\python.exe -m quantum_denoising denoise --image noisy.png --algorithm dequip --out outputs\denoised.npy
.\.venv-qab\Scripts\python.exe -m quantum_denoising denoise --image noisy.png --algorithm qab --threshold 900 --slope 128 --out outputs\qab.npy
```

Ngưỡng QAB 900 ở ví dụ chỉ để minh họa API, chưa được hiệu chỉnh cho ảnh của bạn. De-QuIP mặc định threshold=25 ở đơn vị 0–255, patch=7, radius=10; cần ảnh mỗi chiều >=patch_size. QAB mặc định patch=64 nên cần mỗi chiều >=64. RGB lấy kênh đầu tiên như demo De-QuIP; chưa phải thuật toán denoise màu. Không tự chuẩn hóa hoặc clip dữ liệu trước thuật toán.

## API và cấu trúc

```python
from quantum_denoising import DeQuIPConfig, dequip_denoise, QABImageConfig, qab_image

restored = dequip_denoise(noisy, DeQuIPConfig(threshold=25))
# Chạy thực tế: không cần clean, dùng ngưỡng bạn chọn.
restored = qab_image(noisy, QABImageConfig(threshold=900, slope=128))
# Tái lập source: dùng clean để tìm ngưỡng tốt nhất (oracle).
oracle = qab_image(noisy, reference=clean)
```

```text
src/quantum_denoising/
  core.py        Hamiltonian, eigenbasis, threshold, kiểm tra đầu vào
  qab.py         QAB ảnh + tín hiệu, config bất biến
  dequip.py      Chia patch, tương tác, tái dựng
  noise.py       Nhiễu có seed và SNR chính xác
  metrics.py     PSNR / SNR / SSIM với quy ước tường minh
  io.py          Đọc MAT/NPY/PNG, lưu preview
  benchmark.py   Đo thời gian, xuất dữ liệu, so MATLAB
  cli.py         Giao diện dòng lệnh
tests/           Bản tham chiếu vòng lặp độc lập trên dữ liệu nhỏ
matlab/          Harness gọi hàm MATLAB gốc, không sửa repo
docs/            Ghi chú chuyển đổi và quy trình benchmark
```

## Ý nghĩa kết quả và giới hạn

- QAB source dùng **clean reference để dò ngưỡng theo SNR**. Đó là oracle tuning và có lợi thế so với thuật toán không dùng clean. API tách rõ chế độ oracle và fixed; không trình bày oracle như chạy mù.
- Tối ưu QAB sử dụng trực chuẩn để tính lỗi trong miền hệ số, và chỉ tái dựng nghiệm tốt nhất. Giữ lưới tham số, điều kiện dừng, ngưỡng SNR ban đầu của tác giả. Chi tiết ở [PORTING.md](docs/PORTING.md).
- Benchmark chung dùng PSNR peak=255, SSIM Gaussian 3×3 sigma=1.5, valid/population. Không clip output trước tính metric. JSON còn lưu PSNR/SSIM theo quy ước demo gốc khi khác.
- Thời gian là toàn bộ hàm denoise (gồm chia/ghép patch), không gồm đọc file, tạo nhiễu, metric, hình, import. Ghi cả warmup, các lần đo, median, BLAS backend và số luồng.
- Kiểm thử với bản vòng lặp Python độc lập **không thay thế xác minh trên MATLAB**. Xem [BENCHMARK.md](docs/BENCHMARK.md) để đo và so sai số thực tế. Chưa có MATLAB thì không kết luận Python nhanh/chậm hơn MATLAB.
- QAB ảnh dùng eigen đầy đủ 4096×4096 cho mỗi patch64. Chạy ảnh lớn vẫn tốn CPU/RAM; không có GPU hay xấp xỉ sparse trong bản này.

## Build

```powershell
.\.venv-qab\Scripts\python.exe -m build methods/qab_dequip --no-isolation
# Tạo wheel và sdist trong methods/qab_dequip/dist/.
```

Dùng `requirements-lock.txt` để tái tạo môi trường đã đo trên Windows/Python 3.10. File này khóa dependency; bản Python dự án được cài bằng `pip install --no-deps -e ./methods/qab_dequip` sau đó.

Nguồn: [QAB-denoising](https://github.com/SayantanDutta95/QAB-denoising), [De-QuIP-Denoising](https://github.com/SayantanDutta95/De-QuIP-Denoising). Xem [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) cho commit, tác giả, giấy phép và các bài cần trích dẫn.

