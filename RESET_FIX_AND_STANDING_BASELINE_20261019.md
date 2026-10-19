# Sửa reset cache và kiểm chứng Standing Baseline — 19/10/2026

## Kết quả

**Đã tích hợp bản sửa reset trong project và kiểm chứng bằng PhysX CUDA thực tế
với 1, 8 và 64 environments.** Trong ma trận chính, 2.256 lượt robot auto-reset
sau sửa không có observation vượt sai số cho phép. Actor/critic nhận đúng dữ liệu
trả về từ environment, history không giữ sample của episode đã kết thúc.

**Standing chưa đạt.** Sửa reset không làm checkpoint `model_199.pt` tự biết giữ
thăng bằng. First episode trên 8 robots vẫn khoảng 0,250 s; zero-actions khoảng
1,218 s. Chưa thay posture, PD, action scale, reward, curriculum hoặc PPO config.
**Không chạy training PPO mới. Pilot bên dưới chỉ là đề xuất, chờ người dùng xác nhận.**

Các báo cáo trước được giữ nguyên. Báo cáo này bổ sung bằng chứng triển khai và
test mới cho [chẩn đoán trước](CONTROL_DIAGNOSIS_AND_PLAN_20261019.md), không ghi
đè [kết quả evaluation cũ](EVALUATION_WALKING_BASELINE_20261019.md).

## Root cause và vị trí sửa

Vòng `ManagerBasedRLEnv.step()` chạy physics → termination/reward → reset những
robot ngã → command/events → tính observation và history. Termination/reward đã
đọc root-derived data ở timestamp hiện tại. Các setter reset của PhysX backend
ghi đúng pose/COM velocity mới nhưng không invalidate đầy đủ cache phụ thuộc.
Getter thấy timestamp vẫn hiện hành nên trả angular/linear velocity và projected
gravity cũ. Observation manager reset history đúng về mặt cơ chế, nhưng lấp 5
slots bằng một observation chứa dữ liệu vật lý cũ.

Nguồn đối chiếu tại Lab commit `ffff603eafc6b74264a5261cc0183d6a65390d78`:

- `isaaclab_physx/.../assets/articulation/articulation.py`:
  `write_root_link_pose_to_sim_index`, `write_root_com_velocity_to_sim_index`.
- `articulation_data.py`: derived getters dùng `buffer.timestamp < _sim_timestamp`.
- `isaaclab/envs/manager_based_rl_env.py`: reset và observation cùng một control step.

Thay đổi production chỉ gồm:

1. Thêm [G1MultiSkillEnv](source/g1_datn/g1_datn/tasks/manager_based/g1_multiskill/g1_multiskill_env.py),
   kế thừa `ManagerBasedRLEnv`. Sau `super()._reset_idx(env_ids)`, invalidate timestamp
   8 root-derived buffers: root-link velocity, root-link linear/angular velocity
   trong body frame, root-COM linear/angular velocity trong body frame, root-COM
   pose, projected gravity và heading. Getter tính lại từ source pose/velocity mới.
2. Đổi Gym entry point của `G1-MultiSkill-v0` trong
   [__init__.py](source/g1_datn/g1_datn/tasks/manager_based/g1_multiskill/__init__.py)
   sang lớp project này. Train/play/evaluator dùng `gym.make` đều đi qua bản sửa;
   không cần `--refresh_reset_cache` trong diagnostic nữa.

Không sửa `.deps`, không monkey-patch toàn cục Isaac Lab, không ép observation
về 0, không thêm physics step, không đổi simulation time. Giữ upstream reset
history/manager lifecycle. Cache theo batch nên getter có thể tính lại cả các env
chưa reset; test xác nhận trạng thái vật lý và history của các env đó không bị xóa.

Đây là workaround cho backend/version đã pin, dùng private cache API. Không phải
bản sửa mọi setter của Isaac Lab. Nếu sau này cập nhật backend, phải kiểm tra lại
cache API và chạy regression; tên buffer thiếu sẽ gây lỗi rõ ràng, không bỏ qua âm thầm.

## A. Test observation trước/sau

Script [test_multiskill_reset.py](scripts/rsl_rl/test_multiskill_reset.py) load
checkpoint thật, giữ noise/config training, chạy mean-policy inference. Mỗi nhánh
dùng 2 reset seeds 42/43 × 4 s mô phỏng, 400 control steps. `--baseline` chỉ trong
test bỏ qua override project để tái hiện đường reset cũ. Production luôn bật fix.

Sai số dưới đây là max trên mọi observation audit sau full/manual/auto-reset:

| Env count | Auto-reset robot trước → sau | Max angular error critic trước → sau (rad/s) | Max gravity error critic trước → sau | Vi phạm sau sửa |
| --- | --- | --- | --- | ---: |
| 1 | 30 → 30 | 5,391 → 0 | 0,7611 → 5,96e−7 | 0 |
| 8 | 240 → 248 | 6,155 → 0 | 0,7763 → 5,96e−7 | 0 |
| 64 | 1920 → 1978 | 6,067 → 0 | 0,7794 → 7,15e−7 | 0 |

Số auto-resets giữa nhánh cũ/mới có thể khác vì policy nhận observation khác sau
reset. Đây là số lượt **robot** reset, không phải số batch reset. Angular error
critic đã bỏ scale 0,2. Native reset velocity được assert bằng 0 trong test chuẩn.
Linear-velocity critic trước sửa sai tối đa 0,3265 m/s, sau sửa bằng 0.

Actor có noise trước scale; các giới hạn kiểm tra trên tất cả 5 history slots:

| Actor term | Noise bound sau scale | Max sau sửa trên ma trận chính |
| --- | ---: | ---: |
| Angular velocity | 0,04 | 0,039995 |
| Projected gravity | 0,05 | 0,049999 |
| Relative joint position | 0,01 | 0,010000 |
| Relative joint velocity | 0,075 | 0,074998 |

Command và last-action sau reset khớp chính xác; critic không noise nên kiểm
trực tiếp với native root transform/velocity/joint state, tolerance 1e−5.
Ở reset chuẩn, joint q đúng default và qdot bằng 0. Noise vẫn bật đúng training.

Các kiểm tra bổ sung đã PASS:

- **History:** tất cả term của actor/critic trên reset ids có 5 slots bằng sample
  mới, không còn sample episode cũ. Non-reset ids dịch đúng một slot và giữ 4
  sample trước; không reset toàn batch history.
- **Network consumption:** forward-pre-hook trên MLP actor/critic so trực tiếp
  tensor đầu vào mạng với observation trả về, bao gồm bước tiếp theo sau auto-reset.
  Mỗi nhánh có 401 lượt kiểm tra mỗi mạng. Còn gọi `runner.alg.act(returned_obs)`
  để kiểm đường PPO collection thật; không gọi `learn`, `update`, optimizer hay ghi rollout training.
- **Subset:** nhiều đợt reset chỉ một phần batch, native pose/velocity của robot
  không reset thay đổi đúng 0. Các derived getter của chúng vẫn khớp native state.
- **Setter probes:** pose-only/velocity-only, index/mask, nonzero state, gọi helper
  invalidation project sau write; max angular error 2,09e−6, gravity <1e−6.
  Đây không có nghĩa các setter ngoài reset đã được override toàn cục.
- **Nonzero reset thực qua pipeline:** thêm một event hook tạm trong test 8 envs,
  reset subset với world velocity `[0.2, -0.1, 0.3, 0.7, -0.4, 0.2]` và roll/pitch/yaw
  `(0.17, 0.085, 0.34)` rad. Returned actor/critic observations vẫn đúng cả 5 slots,
  có xét noise, được MLP và `PPO.act` sử dụng. Không clamp về 0. Test này thêm 48
  lượt auto-reset trước probe và kết thúc không lỗi; không đổi config production.

Bằng chứng JSON/log: `run/reset-standing-20261019/reset_{baseline,fixed}_{1,8,64}/`,
`reset_nonzero_8/`; bảng tổng hợp ở [analysis/summary.json](run/reset-standing-20261019/analysis/summary.json).
`--baseline` PASS nghĩa là test đã tái hiện được vi phạm dự kiến, không có nghĩa
pipeline cũ đúng. Nhánh fixed PASS mới yêu cầu không còn vi phạm.

## B. Standing và tầng điều khiển sau sửa

Đã chạy đối chứng trên 1, 8, 64 envs × 3 reset seeds 42/43/44 × 4 mức action gain,
mỗi rollout 2 s. Startup seed 42 cố định; 3 reset seeds không phải 3 independent
training seeds. Với từng env count, giữa các nhánh gain, initial root/joint state,
mass, materials, actor/critic observations khớp trong tolerance 1e−5. Có 9 pairing
checks trong mỗi manifest; toàn bộ PASS. Không dùng diagnostic cache-refresh flag.

First failure tính trước auto-reset, nguyên ngưỡng tilt 0,8 rad và root height 0,2 m:

| Số envs | Model_199 α=1 | α=0,5 | α=0,25 | Zero-actions α=0 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0,2533 s | 0,8800 s | 1,0800 s | 1,2400 s |
| 8 | 0,2500 s | 0,8625 s | 1,0550 s | 1,2175 s |
| 64 | 0,2509 s | 0,8642 s | 1,0588 s | 1,2198 s |

Tất cả first episodes đã kết thúc vì bad_orientation, không có first episode
sống đủ 2 s. Đếm theo từng nhánh: 3, 24, 192 first episodes tương ứng 1/8/64 envs.
α chỉ nhân mean action trong test, không thay action scale chính thức.

[So sánh standing PNG](run/reset-standing-20261019/analysis/standing_comparison.png)
và [trace 0,5 s đầu](run/reset-standing-20261019/analysis/standing_traces.png) có
bản PDF cùng thư mục. Trace dừng ở terminal của first episode; không vẽ observation
sau reset như thể là chuyển động liên tục. CSV summary và từng first episode cùng
NPZ đầy đủ 29 joints/targets/actions/COM/contacts/torques được giữ trong thư mục mới.

### Default posture và PD

Default hai chân đối xứng: hip pitch −0,1; hip roll/yaw 0; knee 0,3; ankle pitch
−0,2; ankle roll 0 rad. Native initial q bằng default; qdot bằng 0. Không phát hiện
sai default offset hay mapping 29 outputs. Target PhysX = default + 0,25 × action,
sai số đo bằng 0; không clip action.

| Nhóm | Kp / Kd | Torque limit Nm |
| --- | --- | ---: |
| Hip pitch/yaw | 100 / 2 | 88 |
| Hip roll | 100 / 2 | 139 |
| Knee | 150 / 4 | 139 |
| Waist yaw | 200 / 5 | 88 |
| Waist roll/pitch | 40 / 5 | 25 |
| Ankle pitch/roll | 40 / 2 | 25 |
| Shoulder/elbow/wrist roll | 40 / 1 | 25 |
| Wrist pitch/yaw | 40 / 1 | 5 |

Mọi actuator implicit; stiffness/damping/force limits đọc ở PhysX khớp asset data.
[CSV mapping và limits đầy đủ](run/reset-standing-20261019/standing_8/gain_1p0/joint_mapping.csv).
Torque log là PD estimate và incoming joint projection, không phải motor torque
sensor thuần. Trong first episodes α=1 và zero không thấy PD estimate vượt effort
limit. α=0,5 có clipping estimate ở right ankle pitch: 7 samples trên cohort 8 envs,
71 trên 64 envs (cộng 3 reset seeds). Không suy rộng thành nguyên nhân mọi lần ngã.

Ví dụ env0/8, seed 42, dữ liệu sau sửa:

| Mốc | Model_199 | Zero-actions |
| --- | --- | --- |
| Action đầu, knee targets L/R | 0,1492 / 0,3853 rad | 0,3 / 0,3 rad |
| Action đầu, ankle pitch targets L/R | −0,5126 / −0,3609 rad | −0,2 / −0,2 rad |
| 0,06 s foot Fz L/R | 411 / 0 N | 1529 / 1479 N |
| 0,10 s pitch; roll | 7,09°; 1,60° | −1,68°; 0,012° |
| 0,26 s pitch; roll | 48,85°; 7,51° — terminal | −1,63°; 0,009° |
| 0,26 s whole-robot COM z | 0,70129 m | 0,70334 m |
| 0,50 s pitch; ankle q L/R | Episode đã kết thúc | 1,88°; −0,3333 / −0,3332 rad |
| Env0 first failure | 0,26 s | 1,24 s |

Zero chưa có contact ở 0,02–0,04 s, touchdown khoảng 0,06 s. COM ban đầu 0,72553 m,
sau 0,06 s còn 0,70893 m: có quá trình rơi/settling ngắn. Đến 0,5 s, zero COM
nghiêng về trước so với midpoint ankles khoảng 0,0799 m (ban đầu 0,0326 m),
ankles lệch target dù targets đối xứng. Đây là proxy hình học; midpoint ankles
không phải support polygon/contact patch đầy đủ.

Đến lúc zero ngã ở 1,24 s: pitch 48,38°, roll 0,108°, COM z 0,51056 m, vx theo
yaw khoảng 1,357 m/s, ankle q L/R −0,727/−0,695 rad trong khi targets vẫn −0,2.
PD estimate peak env0 khoảng 22,15 Nm. Hai chân zero gần đối xứng hơn policy;
không có bằng chứng lỗi đổi tên/tráo khớp gây lệch hai chân.

### Phân loại phần đã biết/chưa biết

- **Lỗi pipeline đã xử lý:** cache state sau reset và observation history nhận dữ
  liệu stale. Test chứng minh fix và việc mạng dùng observation đó.
- **Policy hiện tại chưa giữ thăng bằng:** ngã nhanh ngay first episode sạch;
  mean actions tạo target bất đối xứng và touchdown một chân trước, pitch tăng nhanh.
  Giảm gain kéo dài survival nhưng tất cả vẫn ngã, không phải giải pháp standing.
- **Posture/control chưa được chứng minh tối ưu:** zero giữ nominal joint targets,
  không có feedback COM/balance toàn thân; không thể mặc định nó là controller
  đứng. Settling, loaded ankle error và drift cần ablation riêng. Chưa xác định
  phải đổi posture hay PD; không đổi PD/scale chỉ từ việc thấy zero ngã.
- **Chưa quy toàn bộ cho thiếu PPO iterations:** checkpoint học từ dữ liệu reset
  lỗi; fix runtime không sửa lại weights cũ. Cần pilot sạch để kiểm chứng việc học.

## C. PPO pilot đề xuất — tối đa 256 envs × 200 iterations, CHƯA CHẠY

Mục tiêu đầu là đo tác động của reset fix. Giữ nguyên reward, curriculum, commands,
PD, action scale, std, rollout length và PPO hyperparameters. Khởi tạo mới seed 42,
không resume checkpoint đã học trên pipeline lỗi. Dùng experiment/run directory mới.
24 steps/env × 256 × 200 = 1.228.800 transitions; lưu model_0/100/199 theo save interval
hiện hành. Đây là pilot kiểm reset-corrected learning, chưa phải đổi sang curriculum standing mới.

**Lệnh chỉ để xem xét, không được chạy trước khi người dùng xác nhận:**

```bash
OMNI_KIT_ACCEPT_EULA=YES ./run_multiskill.sh scripts/rsl_rl/train.py \
  --task G1-MultiSkill-v0 --num_envs 256 --max_iterations 200 --seed 42 \
  --experiment_name g1_multiskill_resetfix_pilot --run_name resetfix_seed42_200 \
  --logger tensorboard --visualizer none --kit_first --verify_articulation
```

Đánh giá checkpoint 100/199 của pilot và model_199 cũ trên cùng pipeline đã sửa:

1. Deterministic policy, cùng số envs/seed/randomization; standing và vx=0,1 trước,
   chưa đánh giá thành công ở tốc độ OOD. Giữ ngưỡng terminal gốc.
2. Ít nhất 3 eval seeds, 64 envs, horizon 20 s; báo first-episode và pooled completed
   episodes riêng. Có paired zero/action-gain diagnostic để so tầng điều khiển.
3. Chỉ số chính: survival tại 2/5/10/20 s, thời gian first failure median/p10,
   bad-orientation rate, roll/pitch, COM drift, contact timing, action RMS/rate,
   target error, clipping torque. Tracking XY/yaw RMSE phải đi kèm survival để
   episode rất ngắn không tạo cảm giác tracking tốt.
4. Mức cải thiện pilot mong muốn: median survival tăng ≥2 lần so với old model_199
   trên pipeline đã sửa và ≥50% first episodes sống 5 s. Nếu chưa đạt, báo kết quả
   thất bại và phân tích, không tự kéo dài training. Đây là gate đề xuất, chưa có kết quả.
5. Gate standing để chuyển sang walking curriculum chặt hơn: ≥90% sống đủ 20 s,
   tilt p95 <20° trên các eval seeds. Mean reward/loss giảm không thay thế gate này.
6. Nếu cần đổi standing commands, posture/settling, reward hoặc std thì tạo thí
   nghiệm khác, từng yếu tố một; không gộp vào pilot reset-fix này.

## D. Bảo toàn, giới hạn và tái hiện

Phiên bản thực đo: Python 3.12.13, Sim 6.0.1.0, torch 2.10.0+cu128, RSL-RL 5.0.1;
Lab git tag `v3.0.0-beta2.patch1`, commit không đổi, pip metadata `isaaclab=6.1.14`.
Lab worktree sạch. Hai checkpoint và 51 file bảo vệ từ lượt trước giữ nguyên hash;
trong 27 file snapshot đầu lượt này chỉ Gym registry thay đổi có chủ đích.
Trainer, config, launcher, script chẩn đoán cũ và các báo cáo cũ không đổi.
[Preservation check](run/reset-standing-20261019/preservation_check.json).

File thêm: runtime env project; `test_multiskill_reset.py`;
`summarize_reset_standing.py`; báo cáo này. File sửa: Gym registry nêu trên.
Log/NPZ/JSON/CSV/PNG/PDF mới đều ở `run/reset-standing-20261019/`; không xóa hay
ghi đè log/video/checkpoint/báo cáo cũ. Không commit/push.

Giới hạn còn lại: chưa có policy đứng vững, chưa thử đổi PD/posture, chưa kiểm
balance sau push 5 s, chưa đánh giá rough terrain. Fix chỉ gắn vào reset task này;
setter ngoài reset và backend khác chưa được sửa. Có cảnh báo gRPC/deprecation
nhưng các bài runtime trên đều exit 0. Preflight với EULA được chấp nhận PASS;
đây không phải bằng chứng thay cho physics tests.

Từ `/workspace/Humaniod-MultiSkill`, dùng output mới (regression từ chối ghi đè):

```bash
OMNI_KIT_ACCEPT_EULA=YES ./run_multiskill.sh scripts/rsl_rl/test_multiskill_reset.py \
  --checkpoint logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_199.pt \
  --num_envs 64 --visualizer none --output run/reset-fix-rerun-64

# Thêm --baseline để tái hiện pipeline cũ trong test; đổi output cho mỗi lượt.
# Thay --num_envs bằng 1 hoặc 8 để kiểm các batch sizes còn lại.

OMNI_KIT_ACCEPT_EULA=YES ./run_multiskill.sh scripts/rsl_rl/test_multiskill_reset.py \
  --checkpoint logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_199.pt \
  --num_envs 8 --seconds 0.8 --nonzero_reset_probe --visualizer none \
  --output run/reset-fix-rerun-nonzero

OMNI_KIT_ACCEPT_EULA=YES ./run_multiskill.sh scripts/rsl_rl/diagnose_multiskill_control.py \
  --checkpoint logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_199.pt \
  --num_envs 8 --seeds 42 43 44 --gains 1 0 0.5 0.25 --seconds 2 \
  --visualizer none --output run/standing-resetfix-rerun-8

.venv-multiskill/bin/python scripts/rsl_rl/summarize_reset_standing.py \
  --root run/reset-standing-20261019
```

Hoàn tác production fix: khôi phục đúng file registry từ
`run/reset-standing-20261019/backup/source/g1_datn/g1_datn/tasks/manager_based/g1_multiskill/__init__.py`.
Module runtime mới khi đó không còn được entry point sử dụng; không cần sửa/xóa
framework, checkpoint hay những thay đổi local có sẵn của người dùng.
