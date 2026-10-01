# Chạy G1 đã train bằng FastSAC

Đã hoàn thành 50.000 iterations ngày 01/10/2026. Repository chứa checkpoint
cuối, ONNX, cấu hình và báo cáo đánh giá trong `models/g1_fastsac/`.
Policy học đi theo lệnh vận tốc: tiến/lùi, sang ngang và quay.

## 1. Cài đặt

Cần Linux, GPU NVIDIA với driver hỗ trợ CUDA 12.8, `git` và `uv`.
Để mở viewer cần desktop/X11 và VirtualGL (`vglrun`). Các lệnh dưới đây
được kiểm chứng trên Ubuntu 24.04 với RTX 5060 Ti 16 GB.

```bash
git clone https://github.com/Duckymomo2004/Humaniod.git
cd Humaniod
bash scripts/setup_holosoma.sh
```

Script tạo `.venv-fastsac` riêng với Python 3.11, PyTorch 2.7 CUDA 12.8,
IsaacSim 5.1 và IsaacLab 2.3; tải Holosoma tại commit đã kiểm chứng và áp
dụng bản vá. Lần cài đầu tải nhiều GB. Mỗi script chạy Python thông qua `uv`.

## 2. Chạy thành phẩm

Mở terminal thứ nhất trong thư mục repository:

```bash
bash scripts/evaluate_g1_fastsac.sh models/g1_fastsac/model_0050000.pt --demo --headless
```

Mở terminal thứ hai, cũng trong repository:

```bash
FASTSAC_STATE_FILE=run/fastsac/demo_state.npz bash scripts/view_g1_fastsac.sh
```

Viewer hiển thị chuyển động thật của robot trong IsaacSim thông qua pose stream.
Mặt đất hiển thị là mặt phẳng tham chiếu, không phải bản sao địa hình mô phỏng.
Demo dùng policy đã lưu, không cập nhật trọng số và không ghi trajectory.
Tốc độ được giới hạn 50 Hz; máy chậm có thể chạy dưới thời gian thực.
Lệnh vận tốc do cấu hình đánh giá upstream chọn; demo không chỉ đi thẳng.
Nhấn `Ctrl+C` trong terminal demo và đóng viewer để dừng.

Trên instance hiện tại, demo đã chạy bằng Supervisor. Mở desktop của instance
để xem cửa sổ có nhãn **Final policy demo**. Quản lý bằng:

```bash
sudo supervisorctl status g1-fastsac-demo g1-fastsac-viewer
sudo supervisorctl stop g1-fastsac-demo g1-fastsac-viewer
sudo supervisorctl start g1-fastsac-demo g1-fastsac-viewer
```

Các service Supervisor này chỉ được cài trên instance hiện tại; máy mới dùng
hai terminal ở trên. Script mặc định `DISPLAY=:20`; desktop máy khác có thể
cần đặt `DISPLAY=:0` khi mở viewer.

## 3. Đánh giá policy

```bash
bash scripts/evaluate_g1_fastsac.sh models/g1_fastsac/model_0050000.pt --steps 3000 --headless
```

Đường dẫn `trajectory.npz` được in trong log và nằm trong thư mục evaluation
dưới `logs/fastsac_holosoma/hv-g1-manager/`. Dùng đúng đường dẫn đó:

```bash
uv run --no-project --python .venv-fastsac/bin/python python scripts/fast_sac/summarize_evaluation.py DUONG_DAN/trajectory.npz
```

Đánh giá đã thực hiện: 3.000 bước, tương đương 60 giây mô phỏng. Tất cả kênh
số hữu hạn; RMSE vận tốc tiến 0,230 m/s, ngang 0,242 m/s, quay 0,269 rad/s.
Góc nghiêng thân trung bình 1,77 độ, lớn nhất 4,68 độ. Đây là một lần chạy
với một lệnh vận tốc kết hợp cố định, gồm giai đoạn khởi động; chưa chứng minh
độ bền trên mọi lệnh, địa hình hoặc robot thật. Xem `evaluation_summary.json`.

## 4. Train lại

```bash
bash scripts/train_g1_fastsac.sh --headless
```

Mở viewer cho training trong terminal khác:

```bash
bash scripts/view_g1_fastsac.sh
```

Preset của tác giả: 4.096 môi trường, 50.000 iterations, batch 8.192,
8 updates/bước, learning rate 0,0003, gamma 0,97, tau 0,125, 101 atoms
trên [-20,20], symmetry, chuẩn hóa, compile và BF16. Checkpoint lưu mỗi
1.000 iterations, kèm ONNX và cấu hình. Có thể tiếp tục checkpoint bằng
`--checkpoint DUONG_DAN/model.pt`; replay buffer được thu thập lại.

## 5. Những việc đã làm và giới hạn

- Tích hợp FastSAC chính thức của Holosoma; bổ sung setup và các script chạy bằng `uv`.
- Dùng môi trường Python riêng để giải quyết incompatibility của IsaacSim 6.
- Thử MuJoCo Warp nhưng gặp trạng thái không hữu hạn; chuyển sang IsaacSim.
- Native IsaacSim GUI bị crash trên host này; dùng IsaacSim headless và viewer pose riêng.
- Run đầu gặp metric critic bất thường khoảng iteration 6.800; dừng và resume từ 6.000.
- Thêm kiểm tra categorical target trước update, sao chép metric tensor và warm-up replay khi resume. Nguyên nhân gốc của lỗi đầu chưa xác định.
- Hoàn tất 50.000 iterations, kiểm tra tensor checkpoint hữu hạn và ONNX hợp lệ; chạy đánh giá cuối và mở demo liên tục.

Run recovery mất khoảng 4 giờ 22 phút, chưa tính cài đặt và các lần thử trước.
Bài báo báo cáo 15 phút trên RTX 4090; run này dùng RTX 5060 Ti và backend
IsaacSim khác setup benchmark, cùng các kiểm tra đồng bộ bổ sung, nên không
tái hiện tốc độ đó. Đây là kết quả mô phỏng, chưa triển khai lên robot thật.

Tham khảo: https://younggyo.me/fastsac-humanoid/ và `FASTSAC_RUN.md`.
