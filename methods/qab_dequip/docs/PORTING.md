# Ghi chú chuyển đổi MATLAB → Python

Nguồn được đối chiếu tại các commit sau. Clone tham chiếu mới đặt ở `external/` dưới repo nhóm:

- QAB: `0ec04ac1ae9f5022867f5f004ca67e99507c6c9f`.
- De-QuIP: `384e24f217ea71352fd48ff715fb84b0c4b487f8`.

## Ánh xạ hàm

| MATLAB | Python |
|---|---|
| `image_denoising_QAB` | `qab.qab_image` |
| `signal_denoising_QAB` | `qab.qab_signal` |
| `f_ondes2D`, `f_Hamil2D` | `core.image_basis` |
| `f_ondes1D` | `core.signal_basis` |
| `heavi` | `core.smooth_heaviside` |
| `spatial_patchization` | `dequip.extract_patches` |
| `DeQuIP_denoising` | `dequip.dequip_denoise` |
| `DeQuIP_patch_denoising` | `dequip.denoise_window` |
| `compute_interactions` | `dequip.total_interactions` |
| `reprojection_UWA` (không dịch chuyển) | `dequip.aggregate_patches` |
| `hardthresholding` | ngưỡng strict trong `dequip.denoise_window` |

`image_reconstruction.m` không nằm trong đường chạy demo nên không port; tùy chọn dịch chuyển `xg/yg` của `reprojection_UWA` cũng không nằm trong API này.

## Những chi tiết phải giữ

1. MATLAB `reshape` dùng thứ tự cột. Mọi vector patch trong Python dùng Fortran order; cả thứ tự patch trong cửa sổ tương tác cũng vậy.
2. Hamiltonian ảnh là `diag(potential) + weight*L`: đường chéo của `L` là 4 ở trong, 3 ở cạnh, 2 ở góc. Không nối hàng qua đường biên. Hamiltonian tín hiệu là tuần hoàn, đúng `f_ondes1D`.
3. `sg` là **phương sai** Gaussian, không phải sigma. Gaussian được lấy mẫu trên chiều dài patch rồi FFT tuần hoàn. Không thay bằng `scipy.ndimage.gaussian_filter`, không chuẩn hóa lại tổng kernel.
4. QAB ảnh chia cho `max(patch)`, QAB tín hiệu chia cho `max(abs(signal))`. Tín hiệu/ảnh sạch dùng cùng hệ số. QAB cho ảnh hoàn toàn âm vẫn giữ phép chia này; patch khôngzero nhưng max=0 bị từ chối thay vì sinh NaN.
5. Taper QAB dùng chỉ số 1-based `x = (1:N) - l + 2`; QAB ảnh dò slope `2^linspace(7,11,Ms)`, tín hiệu `2^linspace(1,10,Ms)`. MATLAB `linspace(a,b,1)` trả b; đã xử lý riêng.
6. QAB ảnh giữ bước threshold=12, dừng sau 20 lần liên tiếp kém best SNR trong slope; tín hiệu bước=2 không dừng sớm. Best SNR ban đầu=0 dB như source; nếu không có candidate tốt hơn, trả zero.
7. De-QuIP bỏ mean từng patch trước tính tương tác; cộng mean lại sau tái dựng. Tương tác luôn không âm `p*abs(x[k]-x[n])/distance²`. Tham số Hamiltonian là `fact*max(effective_potential)`.
8. De-QuIP giữ các hệ số có `abs(coeff)>threshold`, lấy `d` eigenvector đầu tiên theo eigenvalue tăng dần. `del` trong hàm MATLAB không được dùng; Python bỏ tham số thừa này.
9. Danh sách tâm cửa sổ MATLAB thêm endpoint kể cả khi đã có. Python cố ý giữ lần lặp trùng vì xóa nó có thể thay trọng số vùng chồng lấp.
   Khi cửa sổ lớn hơn lưới patch (ví dụ ảnh24 với WP7/hW10), giữ cả chỉ số modulo lặp và quy tắc phép gán MATLAB: lần xuất hiện cuối ghi đè kết quả của cùng phần tử; không dùng scatter-add để đếm trùng.
10. Không clip, uint8 hoặc làm tròn output trước metric. PNG preview là bản lượng tử hóa riêng.

## Tối ưu có chủ đích

Hamiltonian thực đối xứng nên dùng `scipy.linalg.eigh(driver='evd')`: eigenvalue tăng dần và eigenvector trực chuẩn. Hệ số QAB được tính `Psi.T @ noisy` thay cho giải hệ tổng quát `linsolve(Psi, noisy)`.

Với ma trận trực chuẩn đầy đủ `Psi`, phép đo lỗi thỏa:

```text
||Psi @ (alpha * taper) - clean||²
  = ||alpha * taper - Psi.T @ clean||²
```

Do đó mỗi candidate chỉ cần thao tác vector; chỉ tái dựng ảnh ứng với candidate thắng. Lưới tìm kiếm và điều kiện dừng vẫn như source. Sai số máy có thể đổi kết quả khi hai candidate gần như hòa; kiểm thử dùng tolerance, không hứa bitwise giống MATLAB. Eigenvector ở eigenspace suy biến có thể khác giữa LAPACK/backend, đặc biệt ảnh hằng và threshold/rank cắt ngang eigenspace.

Tương tác De-QuIP được vector hóa theo batch (mặc định 32 patch), tránh tạo tensor toàn bộ `441×441×49` cùng lúc. Chia/ghép patch cũng được vector hóa. Không thay bằng mô hình học máy, lọc Gaussian đầu ra, GPU, sparse eigensolver hoặc thuật toán khác.

## Phần mở rộng so với source

- API QAB nhận ngưỡng cố định để dùng mà không cần ảnh sạch.
- QAB ảnh nhận patch_size/stride tùy chọn và ảnh chữ nhật, bao phủ mép cuối. Đối chiếu MATLAB nguyên bản phải dùng patch64, stride32, ảnh vuông kích thước chia hết cho32.
- Xử lý đầu vào zero có kết quả hữu hạn; từ chối cấu hình không hợp lệ và dữ liệu NaN/Inf.
- Demo De-QuIP có `file_name` chưa khai báo, nhãn in “BM3D” và `clear all`; harness gọi trực tiếp hàm denoise để tránh phụ thuộc script demo đó.
- Noise QAB được source gọi là Poisson nhưng biểu thức là `randn * sqrt(abs(clean))`, sau đó rescale SNR. CLI gọi đúng là **heteroscedastic Gaussian**, không lấy mẫu Poisson thật.

## Phạm vi xác minh

Các test thực thi bản tham chiếu vòng lặp **Python độc lập** với cách xây ma trận, chia patch và tái dựng trực tiếp giống MATLAB, trên đầu vào nhỏ và không suy biến. Chúng xác minh quy tắc chuyển mã và phép tối ưu, chưa phải parity test với một MATLAB runtime. Hàm MATLAB gốc được để nguyên, và harness đã chuẩn bị để đo sai số/thời gian thực tế khi MATLAB có sẵn.

