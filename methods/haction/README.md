# HAction thử nghiệm

`hamiltonian_repair.py` chứa lớp TensorFlow `HamiltonianAction`: Hf = alpha*(-Delta f) + (V+I)*f, alpha>0. Đây là toán tử thử nghiệm, chưa tái lập phép chiếu theo patch/cơ sở riêng của DIVA.

Các lệnh sau chạy từ thư mục gốc repo, dùng môi trường DIVA:

```powershell
.\.venv-diva\Scripts\python.exe methods/haction/verify_hamiltonian_math.py
.\.venv-diva\Scripts\python.exe methods/haction/audit_hamiltonian.py
.\.venv-diva\Scripts\python.exe methods/haction/run_repair_experiments.py checks --out outputs/haction_checks_run01
```

Kiểm chứng số học không cần tải dataset. Audit dùng notebook DIVA gốc được lưu trong repo; phần xem graph checkpoint chỉ chạy nếu đã tải nguồn DIVA. Kiểm tra tương thích checkpoint lịch sử là tùy chọn và có trạng thái skipped khi thiếu artifact; kiểm chứng toán tử vẫn thực hiện.

## Train nhỏ và chạy mẫu

```powershell
.\.venv-diva\Scripts\python.exe scripts/fetch_upstream.py diva
.\.venv-diva\Scripts\python.exe scripts/download_train_subset.py
.\.venv-diva\Scripts\python.exe methods/haction/run_repair_experiments.py train --steps 300 --out outputs/haction_train_run01
.\.venv-diva\Scripts\python.exe methods/haction/run_hamiltonian_sample.py --checkpoint outputs/haction_train_run01/experimental_H_action_smoke.h5 --out outputs/haction_sample_run01
```

Mạng nhỏ depth4/filter16; 48 train/16 validation; cùng batch nhiễu cho hai biến thể; test Set12. Chọn output mới để giữ các lần đo cũ. Có thể truyền `--train-dir` và `--test-dir` để dùng dữ liệu có sẵn. Runner mẫu cũng nhận `--image`.

Biến môi trường `DIVA_REPO` trỏ đến một clone DIVA khác, `TRAIN400_DIR` trỏ đến thư mục 64 ảnh đã tải; mặc định là `external/DIVA` và `data/Train400_subset64` dưới repo này.

## Đo chi phí train

```powershell
.\.venv-diva\Scripts\python.exe methods/haction/run_repair_experiments.py profile --out outputs/haction_profile_run01
```

Profile đo bước train trên dữ liệu giả lập, không phải một lượt train hội tụ. Lệnh này nặng hơn kiểm tra toán học. Các benchmark lịch sử nằm ở `results/historical`, không bị script sửa.
