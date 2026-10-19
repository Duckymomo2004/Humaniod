# Khôi phục G1 MultiSkill trên RTX 3090 — 2026-10-19

## Kết quả đã chạy thực tế

Đã hoàn thành G1 29 DoF → Isaac Lab → PhysX CUDA → PPO Actor-Critic,
hai iterations với 1 robot, sau đó hai iterations với 64 robot. Hai tiến trình
đều thoát với exit code 0. Không chạy training dài.

| Kiểm tra | 1 robot | 64 robot |
| --- | --- | --- |
| Lệnh | `./run_multiskill.sh --smoke` | `./run_multiskill.sh --smoke64` |
| PPO iterations | 2 | 2 |
| Control steps trên mỗi robot | 48 | 48 |
| Tổng transitions | 48 | 3.072 |
| Backend / device | `PhysxCfg` / `cuda:0` | `PhysxCfg` / `cuda:0` |
| Joint DoF / controlled joints / actions | 29 / 29 / 29 | 29 / 29 / 29 |
| Thời gian phần training, không gồm startup | 2,59 giây | 3,89 giây |
| Checkpoint cuối | `2026-10-19_08-16-14/model_1.pt` | `2026-10-19_08-16-40/model_1.pt` |

Checkpoint nằm dưới `logs/rsl_rl/g1_multiskill_baseline/`. RSL-RL đánh số iterations
từ 0, nên hai iterations là `0/2` và `1/2`. Mỗi run có `model_0.pt`, `model_1.pt`,
TensorBoard events, cấu hình YAML và `smoke_verification.json`. Kiểm tra checkpoint
cho thấy cả actor và critic có trọng số hữu hạn; 9 tensor actor và 8 tensor critic
thay đổi giữa hai checkpoint trong mỗi run. Vì vậy đã có PPO updates thực tế.

## Phiên bản thực tế, không suy ra từ README

| Thành phần | Demo `/workspace/Humaniod` | MultiSkill `/workspace/Humaniod-MultiSkill` |
| --- | --- | --- |
| Branch | `main` | `g1-multiskill` |
| Git HEAD lúc bắt đầu | `fde89319a930185337dfc62b82a12eeb5fff0b38` | cùng commit |
| Python | 3.12.13 | 3.12.13 |
| Isaac Sim / kernel | 6.0.0.1 | 6.0.1.0 |
| Isaac Lab source commit | `28a37cecdd433c22d9eabd6a5954add9f13a8951` | `ffff603eafc6b74264a5261cc0183d6a65390d78` |
| Isaac Lab source tag | `v3.0.0-beta2`, commit được pin trong setup | `v3.0.0-beta2.patch1` |
| Metadata distribution `isaaclab` | 6.1.11 | 6.1.14 |
| PyTorch | 2.10.0+cu128 | 2.10.0+cu128 |
| RSL-RL | 5.0.1 | 5.0.1 |
| USD core wheel | 25.11 | 25.11 |
| Protobuf | 7.36.1 | 7.37.0rc1 |
| grpcio | 1.84.0 | 1.85.0rc1 |

GPU hiện tại: NVIDIA GeForce RTX 3090, 24 GB, driver 595.84.
Unitree HEAD vẫn là `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`, compatibility
patch ban đầu vẫn áp dụng đầy đủ. Không sửa checkout Isaac Lab hoặc Unitree.
Tên phiên bản distribution Isaac Lab khác tên tag source; đã kiểm tra cả hai.

Hai branch vẫn giữ `pyproject.toml` / `uv.lock` của demo, pin Isaac Sim 6.0.0.1 và
`.deps/IsaacLab`. MultiSkill dùng setup riêng và không được `uv sync` từ lockfile đó.
Không chạy setup lại hoặc thay package trong quá trình sửa.

## Nguyên nhân và các phép thử phân biệt

1. Lệnh smoke ban đầu tái hiện `free(): invalid pointer`, cảnh báo pxr trước
   SimulationApp, backtrace `libusd_tf.so` và `libdlfaker.so`, exit 134.
2. Shell thừa hưởng `LD_PRELOAD=libdlfaker.so:libvglfaker.so`, `VGL_ISACTIVE=1`,
   `VGL_DISPLAY=egl` từ desktop. Thử bỏ riêng LD_PRELOAD và VGL_ISACTIVE trong
   tiến trình con vẫn gặp invalid pointer tại USD, lần đó exit 139. Do đó không
   thể quy nguyên nhân chính cho VirtualGL.
3. Import Torch, RSL-RL và các package đăng ký task chưa nạp pxr. Khi resolve
   config MultiSkill, wildcard MDP imports của Unitree kéo runtime action classes
   → Articulation → SimulationContext → SceneDataProvider → pxr. `/proc/self/maps`
   lúc đó chứa `usd_core.libs/libusd_ms-79e18534.so`. Kit khởi động sau đó nạp
   bộ USD riêng; lỗi native xảy ra tại registry `TfEnum::_AddName`.
4. Phép thử chỉ SimulationApp khởi động thành công. Offline config demo cũ
   `G1DatnEnvCfg` không nạp pxr (`PXR loaded: False`), khác config MultiSkill.
5. Bật Kit trước Hydra/config giải quyết native crash. Đặt bootstrap sau các
   import cấu hình Lab vẫn gặp `Unknown asset config type for height_scanner`:
   AppLauncher tạm reload Lab modules, làm xuất hiện hai class identity
   SensorBaseCfg. Bản cuối khởi động trước mọi import cấu hình Lab, xử lý cả hai lỗi.
6. Sau khi startup chạy được, phát hiện USD asset vẫn là placeholder
   `path/to/unitree_model/...`. Đã tải đúng 4 file G1 29 DoF từ dataset Unitree,
   không thay robot bằng asset demo cũ.
7. Hai smoke cuối chạy thành công với VirtualGL kế thừa còn nguyên và vẫn có
   cảnh báo duplicate `grpc/health/v1/health.proto`. Cảnh báo đó không làm dừng
   những lần chạy đã kiểm chứng; không đổi Protobuf/grpcio để che cảnh báo.

Đây là xung đột nạp thư viện native do import/startup order. Không có bằng chứng
cần hạ Isaac Sim, đổi tag Isaac Lab, đổi Torch hoặc thay toàn bộ dependency.

## Những gì tái sử dụng từ demo

- Dùng Python đúng venv và `ISAACLAB_PATH` đúng checkout tương ứng.
- Giữ Python 3.12, Torch cu128, RSL-RL 5.0.1 và `OMP_NUM_THREADS=4` mặc định.
- Dùng viewer `none` cho bài kiểm tra nền tảng, không cần dựng hình desktop.
- `PYTHONDONTWRITEBYTECODE=1` tránh ghi thêm bytecode khi chạy.

Không sao chép lockfile/Isaac Sim/tag Isaac Lab cũ sang MultiSkill. Demo dùng asset
G1 từ NVIDIA và có Newton Viewer cho hiển thị; MultiSkill cần asset G1 29 DoF của
Unitree. Không đổi physics sang Newton để tránh Kit.

## File thay đổi và khả năng hoàn tác

- `run_multiskill.sh`: export đường dẫn model riêng; truyền `--kit_first` và
  `--verify_articulation` cho smoke; tắt ghi bytecode. Quyền executable đã có từ trước.
- `scripts/rsl_rl/train.py`: bootstrap AppLauncher sớm theo cờ, quản lý đóng app
  và exit code; in/kiểm tra articulation và lưu kết quả sau PPO.
- `source/g1_datn/g1_datn/tasks/manager_based/g1_multiskill/g1_multiskill_env_cfg.py`:
  chỉ thay placeholder asset bằng đường dẫn từ `UNITREE_MODEL_DIR`.
- `scripts/fetch_multiskill_asset.py`: mới; tải 4 file chính thức, pin revision
  `323e350252b9c3aee9c40acbfdad84f6ce46a5ac`, kiểm tra hash, không ghi đè file đã có.
- `README_MULTISKILL_VM.md` và báo cáo này: bổ sung cách chạy và bằng chứng.
- Thêm `.deps/unitree_model-multiskill/` (~28 MB), logs và checkpoint của các bài thử.

Không thay reward, curriculum, actuator settings hoặc thiết kế kỹ năng.
Không thay package, không xóa venv/deps/checkpoint, không sửa môi trường demo,
không sửa cấu hình hệ thống, không commit hoặc push.
File bytecode `scripts/rsl_rl/__pycache__/cli_args.cpython-312.pyc` đã dirty từ
trước khi bắt đầu, được giữ nguyên.

Bản gốc `run_multiskill.sh` và `train.py` đã sao lưu trong
`run/startup-recovery-20261019/`. Config gốc được lưu ở đó với tên
`g1_multiskill_env_cfg.original.py`. Để hoàn tác thay đổi code, sao chép các bản
gốc về đúng vị trí; không cần xóa môi trường hoặc checkpoints. File mới có thể
giữ lại mà không ảnh hưởng runtime khi launcher gốc được phục hồi.

## Chạy lại

```bash
cd /workspace/Humaniod-MultiSkill
export OMNI_KIT_ACCEPT_EULA=YES
./run_multiskill.sh --smoke
./run_multiskill.sh --smoke64
```

Asset đã có sẵn. Nếu chuyển bản sửa sang một working tree mới với môi trường
MultiSkill đúng phiên bản, tải asset một lần bằng:

```bash
.venv-multiskill/bin/python -B scripts/fetch_multiskill_asset.py
```

Nếu đã có dataset tại nơi khác, export `UNITREE_MODEL_DIR=/duong/dan/unitree_model`
trước khi chạy. Với lệnh gọi trainer trực tiếp thay vì shortcut smoke, cần thêm
`--kit_first`. Không tự chạy training 3.000 iterations.

## Articulation đọc trực tiếp từ simulator

29 joint DoF dưới đây không tính 6 bậc tự do floating base. Thứ tự dưới đây là
thứ tự simulator trả về, không phải thứ tự SDK Unitree:

```text
left_hip_pitch_joint
right_hip_pitch_joint
waist_yaw_joint
left_hip_roll_joint
right_hip_roll_joint
waist_roll_joint
left_hip_yaw_joint
right_hip_yaw_joint
waist_pitch_joint
left_knee_joint
right_knee_joint
left_shoulder_pitch_joint
right_shoulder_pitch_joint
left_ankle_pitch_joint
right_ankle_pitch_joint
left_shoulder_roll_joint
right_shoulder_roll_joint
left_ankle_roll_joint
right_ankle_roll_joint
left_shoulder_yaw_joint
right_shoulder_yaw_joint
left_elbow_joint
right_elbow_joint
left_wrist_roll_joint
right_wrist_roll_joint
left_wrist_pitch_joint
right_wrist_pitch_joint
left_wrist_yaw_joint
right_wrist_yaw_joint
```

Actor MLP: 480 → 512 → 256 → 128 → 29, Gaussian distribution.
Critic MLP: 495 → 512 → 256 → 128 → 1.
Control dt 0,02 giây, physics dt 0,005 giây, decimation 4.

## Bằng chứng local và giới hạn

Trong `run/startup-recovery-20261019/`:

- `environment-comparison.json`: metadata phiên bản và commit thực tế.
- `smoke-original.log`: lỗi native ban đầu.
- `smoke-no-vgl.log`: bỏ VirtualGL vẫn lỗi.
- `import-audit.log`, `import-stack.log`: import pxr và thư viện USD đã nạp.
- `demo-config-import.log`: config demo không nạp pxr.
- `native-library-verified.log`, `native-library-verified.exitcode`: chỉ Kit USD
  được nạp, class sensor đúng identity, CUDA Torch hoạt động, VirtualGL vẫn hiện
  trong maps, exit 0. Phép dò ban đầu `native-library-audit.log` có lỗi kiểm tra
  `pxr.__file__` (namespace package trả None); dùng bản verified đọc `Usd.__file__`.
- `smoke-kit-first.log`: vượt native startup, còn thiếu asset.
- `smoke-asset.log`: lỗi sensor class identity khi bootstrap vẫn quá muộn.
- `smoke-early-bootstrap.log`: 1 robot chạy đủ 2 iterations.
- `smoke64.log`, `smoke64.exitcode`: 64 robot chạy đủ 2 iterations, exit 0.
- `checkpoint-audit.json`: trọng số hữu hạn và thực sự thay đổi trong cả hai run.
- `asset-fetch.log`: file asset được xác minh hash.
- `preflight-verified.log`: preflight sau sửa, với EULA đã được chấp nhận.

Còn cảnh báo Protobuf/gRPC duplicate descriptor, MDP/PPO configuration deprecated,
obs_groups fallback, MaterialX/visualizer extension và TGS solver. Không cần đổi
dependency hay solver để hoàn thành hai smoke. Chưa kiểm chứng viewer, running,
jumping, recovery, chất lượng đi bộ hoặc training dài hạn. Hai iterations chỉ
kiểm tra nền tảng mô phỏng và cập nhật PPO.
