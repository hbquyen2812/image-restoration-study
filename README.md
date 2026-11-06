# Image Restoration Study

Repo nghiên cứu nhóm về khử nhiễu ảnh bằng phương pháp thông thường và quantum-inspired. Code hiện có chạy trên CPU; chưa có mạch lượng tử/QPU.

## Code đã có

| Thành phần | Vị trí | Trạng thái |
|---|---|---|
| QAB, De-QuIP Python | [methods/qab_dequip](methods/qab_dequip) | API, CLI, benchmark, 36 kiểm thử, harness MATLAB |
| DIVA checkpoint | [methods/diva/run_checkpoint.py](methods/diva/run_checkpoint.py) | Chạy checkpoint tác giả, xuất metric và ảnh |
| Mã DIVA gốc | [methods/diva/upstream](methods/diva/upstream) | Notebook và MATLAB helper gốc, giữ license |
| HAction | [methods/haction](methods/haction) | Toán tử thử nghiệm, audit, kiểm chứng số học, train nhỏ |
| Kết quả lịch sử | [results](results) | Số đo cũ, không phải lần benchmark mới |
| CNN baseline độc lập | [methods/baselines](methods/baselines) | Chưa triển khai |
| Pipeline chung | [common](common) | Chưa hợp nhất; mỗi phương pháp đang giữ protocol riêng |

HAction chưa tái lập chính xác kiến trúc DIVA. QAB benchmark có chế độ oracle dùng ảnh sạch để chọn ngưỡng. Chưa hoàn thành benchmark chung trên DIV2K, Set14, BSD100 và Urban100.

## Cài đặt

Chạy các lệnh dưới đây từ thư mục gốc repo, bằng PowerShell và Python 3.10. Hai môi trường tách riêng vì DIVA dùng TensorFlow/NumPy cũ.

```powershell
py -3.10 -m venv .venv-qab
.\.venv-qab\Scripts\python.exe -m pip install -r methods/qab_dequip/requirements-lock.txt
.\.venv-qab\Scripts\python.exe -m pip install --no-deps -e ./methods/qab_dequip
.\.venv-qab\Scripts\python.exe -m pytest -q methods/qab_dequip/tests
.\.venv-qab\Scripts\python.exe -m quantum_denoising --help

py -3.10 -m venv .venv-diva
.\.venv-diva\Scripts\python.exe -m pip install -r methods/diva/requirements.txt
.\.venv-diva\Scripts\python.exe methods/haction/verify_hamiltonian_math.py
```

Lần kiểm tra toán học trên clone mới bỏ qua kiểm tra tương thích checkpoint lịch sử nếu chưa có hai artifact cũ; trạng thái này được ghi rõ trong JSON.

## Tải nguồn tham chiếu và chạy

Dataset, checkpoint và repo tham chiếu tải riêng vào thư mục bị Git bỏ qua. Script ghim commit trong [configs/upstream_sources.json](configs/upstream_sources.json).

```powershell
.\.venv-diva\Scripts\python.exe scripts/fetch_upstream.py diva
.\.venv-diva\Scripts\python.exe methods/diva/run_checkpoint.py --sigma 25 --dataset Set12 --out outputs/diva_sigma25_run01

.\.venv-qab\Scripts\python.exe scripts/fetch_upstream.py qab dequip
.\.venv-qab\Scripts\python.exe -m quantum_denoising benchmark --image external/De-QuIP-Denoising/De-QuIP/data/cameraman.png --sizes 64 --repo-root external --out outputs/qab_dequip_run01
```

Chọn thư mục output mới cho mỗi lần benchmark. Chỉ chuyển các bảng/hình cần chia sẻ vào `results/` sau khi xem lại; `outputs/`, `data/`, `checkpoints/`, `external/` và môi trường Python không được commit.

## HAction: audit và huấn luyện chẩn đoán

```powershell
.\.venv-diva\Scripts\python.exe methods/haction/audit_hamiltonian.py
.\.venv-diva\Scripts\python.exe methods/haction/run_repair_experiments.py checks --out outputs/haction_checks_run01
.\.venv-diva\Scripts\python.exe scripts/download_train_subset.py
.\.venv-diva\Scripts\python.exe methods/haction/run_repair_experiments.py train --steps 300 --out outputs/haction_train_run01
```

Train cần nguồn DIVA đã tải để lấy Set12. Subset chẩn đoán gồm 48 ảnh train và 16 validation, tách theo ảnh gốc. Đây không phải huấn luyện đầy đủ trên DIV2K.

Xem [hướng dẫn HAction](methods/haction/README.md) cho đường dẫn tùy chọn, chạy mẫu và đo thời gian; [làm việc nhóm](docs/CONTRIBUTING.md) cho quy trình Git.

## Nguồn và quyền tác giả

- DIVA: giữ mã gốc và [MIT license](methods/diva/upstream/LICENSE), commit `91a1a847500ce43151f243a82dabff8cd0db7c6d`.
- QAB/De-QuIP: xem [THIRD_PARTY_NOTICES](methods/qab_dequip/THIRD_PARTY_NOTICES.md). Không áp dụng license QAB cho nguồn De-QuIP.
- [migration_sources.json](docs/migration_sources.json) ghi nguồn và hash trước khi sửa đường dẫn; không phải checksum của file sau chỉnh sửa.

Repo lưu các bản triển khai nghiên cứu của nhóm, không phải release chính thức của tác giả các phương pháp.
