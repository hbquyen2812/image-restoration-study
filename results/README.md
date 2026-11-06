# Kết quả lịch sử

Các JSON trong `historical/` sao chép số đo đã có trước khi chuyển code sang repo này, không phải các lượt chạy mới. Đường dẫn và metadata bên trong mô tả workspace gốc.

| File | Ý nghĩa |
|---|---|
| `diva_set12_sigma25.json` | 12 ảnh Set12, Gaussian sigma25, checkpoint tác giả: PSNR 30.4934 dB, SSIM 0.8639; tổng forward 37.8357 s ở lượt lưu này |
| `paired_training.json` | 300 bước: mã gốc 25.0251 dB; HAction 23.9407 dB; một seed, mạng nhỏ |
| `training_profile.json` | Thời gian từng bước và ngoại suy; chưa full train |
| `hamiltonian_math_checks.json` | Kiểm chứng toán học lịch sử, gồm tương thích checkpoint cũ |
| `cameraman.json` | QAB-oracle/De-QuIP, ảnh resize64/128, Gaussian SNR22 |
| `qab_sample.json`, `qab_signal.json` | Ảnh/tín hiệu mẫu với cấu hình và thang metric riêng |

Không xếp hạng chéo các nhóm: khác dữ liệu, noise, SSIM và số luồng CPU. QAB-oracle dùng ảnh sạch để chọn ngưỡng. Đạt kiểm thử Python không chứng minh tương đương MATLAB runtime.

Kết quả mới mặc định ghi vào `outputs/` bị Git bỏ qua. Sau khi kiểm tra protocol, chọn bảng/hình cần chia sẻ và ghi rõ cấu hình, commit, seed, phần cứng, phạm vi so sánh trước khi thêm vào đây.
