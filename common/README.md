# Pipeline dùng chung — chưa triển khai

Hiện QAB/De-QuIP và DIVA/HAction giữ bộ sinh nhiễu và metric riêng trong mã phương pháp. Chưa được phép coi các bảng lịch sử là benchmark chung.

Việc tiếp theo: thống nhất đọc ảnh, grayscale/RGB, Gaussian/Poisson-Gaussian, clipping, seed, PSNR và cấu hình SSIM; sau đó mới chuyển các hàm dùng chung vào đây. Phép thử Poisson-Gaussian trước đây chạy trong phiên làm việc, chưa có script độc lập được chuyển sang repo này.
