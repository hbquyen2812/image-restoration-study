# Làm việc nhóm

1. Clone repo, đọc README và chọn đúng môi trường cho phương pháp.
2. `git switch main`, `git pull --ff-only`, rồi tạo nhánh như `git switch -c feat/noise-pipeline`.
3. Mỗi thay đổi ghi rõ mục tiêu, nguồn thuật toán và cách chạy. Không đổi dữ liệu test để chọn tham số.
4. Chạy kiểm tra phù hợp: QAB/De-QuIP dùng pytest; HAction dùng verify_hamiltonian_math.py; DIVA dùng một ảnh trước benchmark đầy đủ.
5. Xem `git status --short` và `git diff --cached --stat` trước commit. Không đưa môi trường Python, dataset, checkpoint, cache hoặc toàn bộ repo upstream vào Git.
6. Push nhánh, mở pull request vào main và nhờ một thành viên xem code/kết quả.

Các thư mục đang chờ triển khai được đánh dấu trong README; sự tồn tại của thư mục không đồng nghĩa đã có mô hình hoặc benchmark ở đó.
