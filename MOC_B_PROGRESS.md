# DATN G1 29 DoF - Tiến độ Mốc B
Ngày: 19/10/2026

## 1. Các mốc đã hoàn thành

- Mốc 0: Isaac Lab + PhysX CUDA + G1 29 DoF.
- PPO smoke test: 64 environments, 2 iterations - PASS.
- PPO pilot: 256 environments, 200 iterations.
- Đã lưu model_100.pt và model_199.pt **trên máy Vast.ai** (chưa xác nhận đã sao lưu checkpoint lên GitHub).
- Mốc A: Đã sửa và kiểm chứng observation cache sau reset.

## 2. Mốc B - Posture, Settling, PD

Trạng thái: Đang thực hiện, chưa kết luận cuối cùng.

Codex đã chạy script:
`scripts/rsl_rl/summarize_posture_pd.py`

Thư mục kết quả trên máy Vast.ai:
`run/posture-pd-20261019/analysis`

Các file phân tích đã ghi nhận:
- `summary.json`, `standing_summary.csv`, `joint_metrics.csv`
- `joint_setup.csv`, `body_masses.csv`, `zero_env0_timeseries.csv`
- `comparison.png`, `pd_sensitivity.png`, `support_polygons.png`, `zero_dynamics.png`
- Các biểu đồ tương ứng ở định dạng PDF.

Log gần nhất ghi nhận:
- Kiểm tra hard joint bound excess > 0.001 rad: 0 trong dữ liệu được kiểm tra.
- Chưa có báo cáo nguyên nhân cuối cùng.

## 3. Kết quả từ trước Mốc B

- Policy model_199: mất thăng bằng sau khoảng 0,25 s.
- Zero-actions: khoảng 1,22 s.
- Giảm action xuống 50%: khoảng 0,863 s.
- Giảm action xuống 25%: khoảng 1,055 s.

## 4. Những việc chưa hoàn thành

- Tổng hợp đầy đủ kết quả Mốc B.
- Kết luận về posture, COM, support polygon.
- Kết luận về PD gain và giới hạn torque.
- Xác định nguyên nhân chính khiến robot mất thăng bằng.
- PPO pilot sau sửa reset.

## 5. Bước tiếp theo

1. Đọc kết quả có sẵn trong `run/posture-pd-20261019`.
2. Kiểm tra các thay đổi local bằng `git diff`.
3. Tổng hợp nguyên nhân từ dữ liệu, không chạy lại thí nghiệm nếu dữ liệu hiện có đã đủ.
4. Chỉ sau khi chẩn đoán xong mới quyết định PPO pilot mới.

**Lưu ý:** Chưa huấn luyện Walking Baseline chính thức. Không ghi đè checkpoint cũ.

## 6. Trạng thái sao lưu

File ghi chú này đã được lưu trực tiếp lên branch `g1-multiskill` trên GitHub. **Mã nguồn đang sửa, dữ liệu CSV/PNG và checkpoint từ VM cần được đẩy lên riêng bằng `git push` hoặc phương thức sao lưu khác; chưa được xác nhận đã đồng bộ.**
