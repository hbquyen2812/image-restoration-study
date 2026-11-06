# DIVA

`run_checkpoint.py` chạy checkpoint tác giả trên Set12/BSD68 và xuất JSON, CSV, NPZ, PNG. Các notebook trong `upstream/` được giữ nguyên từ commit được ghi ở README gốc; chúng là mã tham chiếu, có đường dẫn/môi trường lịch sử. Dùng runner cho lần chạy có kiểm soát.

Từ thư mục gốc repo:

```powershell
.\.venv-diva\Scripts\python.exe scripts/fetch_upstream.py diva
.\.venv-diva\Scripts\python.exe methods/diva/run_checkpoint.py --dataset Set12 --sigma 25 --out outputs/diva_run01
```

Đã có clone DIVA ở nơi khác thì truyền `--repo "duong-dan-clone-DIVA"`. Runner đọc checkpoint/data ở clone đó và ghi commit Git nguồn vào báo cáo. Dùng `--limit 1` để kiểm tra một ảnh.

Tham số mặc định: Gaussian sigma=25 ở thang 0–255, seed=0, CPU 4 luồng; tính metric trên mảng float chưa clip, SSIM data_range=1. Chỉ PNG preview được clip.

Checkpoint/ảnh không được đưa vào repo nhóm. Chưa có runner cho super-resolution hoặc phép so sánh huấn luyện đầy đủ với CNN baseline. Phần Hamiltonian thử nghiệm ở `methods/haction`, tách khỏi mã DIVA gốc.
