# G1 MultiSkill — Ghi chú kiểm thử 29 DoF (2026-10-19)

> **Trạng thái: PASS — 5/5 tiêu chí smoke test.** Đây là kết quả kiểm thử pipeline mô phỏng/huấn luyện PPO, **không** phải bằng chứng policy đã học đi bộ ổn định hoặc có các kỹ năng MultiSkill.

## 1. Phạm vi và nguồn bằng chứng

- Repository: `Duckymomo2004/Humaniod`, branch `g1-multiskill`.
- Task: `G1-MultiSkill-v0`.
- Kết quả được ghi lại từ **log terminal người thực hiện cung cấp** ngày 2026-10-19:
  - Một môi trường: run `2026-10-19_08-22-10`.
  - 64 môi trường: run `2026-10-19_08-25-40`.
- File này tóm tắt kết quả đã quan sát, không đính kèm bản gốc của toàn bộ log terminal.

## 2. Cấu hình

| Thành phần | Giá trị |
|---|---|
| Máy chạy | Vast.ai, Ubuntu 24.04, NVIDIA RTX 3090 (24 GiB VRAM) |
| Python | 3.12 |
| Isaac Sim mục tiêu | `6.0.1.0` |
| Isaac Lab mục tiêu | `v3.0.0-beta2.patch1` |
| RSL-RL mục tiêu | `5.0.1` |
| PyTorch mục tiêu | `2.10.0+cu128` |
| Unitree RL Lab | `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` + compatibility patch của repo |
| Physics backend ghi nhận | `PhysxCfg`, `cuda:0` |
| Physics step | `0.005 s` |
| Environment/control step | `0.02 s` |

**Chú ý:** Các version ghi “mục tiêu” dựa trên `README_MULTISKILL_VM.md` và `setup_multiskill.sh`; muốn ghi nhận bản cài thực tế của một máy mới, cần xuất lại `pip freeze`/kiểm tra tag và commit.

## 3. Robot và PPO thực tế đã khởi tạo

- Articulation: **29 DoF**.
- Khớp được điều khiển: **29**.
- Action manager: `JointPositionAction`, **29 chiều**, khớp index 0–28.
- Actor observations: **480 chiều**, MLP `480 → 512 → 256 → 128 → 29`, kích hoạt ELU; Gaussian action distribution.
- Critic observations: **495 chiều**, MLP `495 → 512 → 256 → 128 → 1`, kích hoạt ELU.
- Config có **19 reward terms**, **2 curriculum terms** và command `base_velocity` kế thừa baseline.

## 4. Checklist smoke test

- [x] Isaac Lab đăng ký và load được `G1-MultiSkill-v0`.
- [x] Spawn và khởi tạo articulation G1 **29 DoF**.
- [x] Khởi tạo **64 environments** song song.
- [x] Khởi tạo Actor và Critic.
- [x] Hoàn thành **2 PPO iterations với 64 environments**, không dừng vì crash.

### Kết quả các lần chạy

| Bài thử | Envs | PPO iterations | Tổng transitions ghi nhận | Kết quả |
|---|---:|---:|---:|---|
| `./run_multiskill.sh --smoke` | 1 | 2/2 | 48 | PASS |
| `./run_multiskill.sh --smoke64` | 64 | 2/2 | **3.072** | **PASS** |

Bài thử 64 environments:
- `24` steps/env/iteration × `64` envs × `2` iterations = **3.072 transitions**.
- Tốc độ log khoảng **857–974 steps/s**.
- `Training time: 3.97 seconds` (thời gian training được báo bởi ứng dụng, không phải tổng thời gian setup và startup).
- Mean reward: `-0.68` (iteration 0), `-4.03` (iteration 1).
- Mean episode length cuối: `39.31`; `Metrics/success_rate` cuối: `0.0000`.

Log xác nhận:
```text
[PASS] PhysX CUDA + G1 29 DoF + PPO:
{"num_envs": 64, "physics": "PhysxCfg", "device": "cuda:0",
 "num_dof": 29, "num_controlled_joints": 29,
 "action_dim": 29, "ppo_iterations": 2, "physics_control_steps": 48}
```

**Diễn giải đúng:** 2 iterations xác minh **pipeline** và cập nhật PPO hoạt động. Chưa có kết luận về chất lượng walking, hội tụ, độ bền, recovery, running hoặc jumping. Checkpoint `.pt` không được bảo đảm có sau 2 iterations (runner có thể chỉ lưu theo `save_interval`).

## 5. Tái lập bài thử

Trong máy đã cài đầy đủ môi trường theo hướng dẫn `README_MULTISKILL_VM.md`:

```bash
cd /workspace/Humaniod-MultiSkill
./run_multiskill.sh --preflight

# Sau khi đọc và chấp nhận NVIDIA EULA:
export OMNI_KIT_ACCEPT_EULA=YES
./run_multiskill.sh --smoke
./run_multiskill.sh --smoke64
```

Script thiết lập: `bash setup_multiskill.sh` (chỉ khi cần tạo/cài lại môi trường). Không dùng `uv.lock` dành cho demo cũ để cài MultiSkill.

## 6. Log và việc sao lưu

Các thư mục log ghi nhận trên máy ảo:
- `logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-22-10/`
- `logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-25-40/`

**Tài liệu này được lưu trên GitHub, nhưng không có nghĩa các log/patch/checkpoint chỉ nằm ở VM đã được đồng bộ lên GitHub.** Trước khi hủy instance Vast.ai, cần kiểm tra `git status` và sao lưu riêng các file cần giữ. Không commit `.venv*`, `.deps`, cache, crash dump hoặc file chứa credential. Với log cần đưa lên GitHub, chỉ chọn file nhỏ đã kiểm tra nội dung; GitHub không phù hợp để lưu mọi checkpoint lớn.

## 7. Công việc tiếp theo

1. Lưu/đối chiếu các sửa đổi local của Codex vào GitHub sau khi review.
2. Kiểm thử walking baseline lâu hơn, đánh giá reward/velocity tracking và lưu checkpoint.
3. Thiết kế các thành phần MultiSkill thật: gait-conditioned commands, reward routing, curriculum, transitions, running, jumping và fall recovery.
