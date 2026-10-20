# Mốc C — Kế hoạch PPO pilot và cổng phê duyệt

Ngày lập: **20/10/2026**, instance **56912565**, repository **Humaniod-MultiSkill**.
**STAGE 8: PASS WITH LIMITATIONS — THIẾT KẾ OFFLINE.**
**NOT READY — SPECIFIC BLOCKERS REMAIN.**
**TRAINING BACKUP GATE — NOT SATISFIED.**

Tài liệu này không phê duyệt hoặc thực thi training. Toàn bộ command/diff bên dưới chỉ để xem xét trong bước tiếp theo; không được thực thi chỉ vì được ghi trong kế hoạch.

## 1. Executive summary

Đề xuất **C1: fresh initialization, 256 env ×200 PPO iterations, seed 42, 24 rollout steps/env**, tổng **1.228.800 transitions**. Dùng task G1-MultiSkill-v0 với G1MultiSkillEnv và reset fix hiện có, PhysX CUDA, TensorBoard. Giữ reward, curriculum, PD, posture, action scale và mạng hiện hành. Chỉ thay số env, iteration cap, experiment/run name và logging directory qua CLI; không resume model_199.

Mục tiêu: xem một policy mới học trên pipeline reset-corrected có đạt tiến bộ định lượng hay không. Chưa có nguyên nhân để kéo dài training hoặc đổi nhiều thành phần đồng thời. C1 **không mặc nhiên là A/B chỉ khác reset so với run ngày 19/10** do thiếu historical YAMLs/manifests và khác máy.

Chọn checkpoint cuối model_199 làm candidate chính; model_0 là early-policy baseline (đã qua **1 PPO iteration**, không phải weights chưa train), model_100 là diagnostic trung gian. Evaluate cả ba cùng zero-actions trên 64 env, seeds 42/43/44, horizon 20s, noise/randomization/termination được giữ. Không dùng reward tăng thay bằng chứng standing.

Những việc phải giải quyết trước execution: approved external backup destination/policy; hourly rate hoặc approved cost/time cap; chấp thuận metrics/gates và evaluation-instrumentation proposal. Telemetry contact quality chưa đạt chỉ chặn việc dùng support margin làm acceptance metric chính, không nhất thiết chặn C1 dùng survival/termination.

## 2. Verified research baseline

| Nguồn | Kết quả đã có | Ý nghĩa và giới hạn |
|---|---|---|
| [Stage5](.deps/stage5-logs/verification-20261020T172009Z.json) | Smoke1/64 ×2iterations, finite/changed actor và critic, PhysX CUDA29DoF | Runtime/update smoke, không chứng minh 256env hoặc 200iterations |
| [Stage6](.deps/stage6-logs/REPORT.md) | Fixed8/64:432robot auto-resets,0violations; negative control14events | Reset regression trên smoke checkpoint mới, không phải lịch sử model199 |
| [MốcB](MOC_B_FINAL_REPORT.md) | 18cases ×24first episodes,51pairing;432episodes đều ngã,0 sống 5s | Aggregate lịch sử,18 raw dirs còn thiếu |
| MốcB reference | Policy α1=0,2500s; zero=1,2175s; Kp2/4/8=1,4892/2,4150/2,6892s | Kp4 có 193 estimated clipping samples; Kp8 SD 0,3561s so Kp4 0,0375s; chưa stable standing |
| [Recovery lịch sử](MULTISKILL_RECOVERY_REPORT.md) | Early Kit bootstrap và official asset giải quyết startup | Số liệu/driver/ngày cũ không phải benchmark máy mới |

Đã đọc [MOC_B_PROGRESS](MOC_B_PROGRESS.md), [CONTROL_DIAGNOSIS_AND_PLAN](CONTROL_DIAGNOSIS_AND_PLAN_20261019.md), [RESET_FIX_AND_STANDING_BASELINE](RESET_FIX_AND_STANDING_BASELINE_20261019.md) và [EVALUATION_WALKING_BASELINE](EVALUATION_WALKING_BASELINE_20261019.md) cùng source. Những kết luận lịch sử như completed-episode mean0,2597s không được trộn với first-episode mean0,25s. MốcB phát hiện t=0 onset đáng nghi; không dùng nó làm gate PPO.

## 3. Training configuration provenance

### 3.1 Nguồn cấu hình và dấu vân tay

HEAD **fc46f39ef0214e84cb0bb484dfec0d1fef7ba161**, branch **g1-multiskill**. IsaacLab source **ffff603eafc6b74264a5261cc0183d6a65390d78**, tag v3.0.0-beta2.patch1; distribution metadata 6.1.14. Unitree source **4960b84732b0c2ec593dccbfe963fda1bcd7b1e3** với compatibility patch hiện có.

Source chain:
[G1 env](source/g1_datn/g1_datn/tasks/manager_based/g1_multiskill/g1_multiskill_env_cfg.py) kế thừa Unitree RobotEnvCfg, override official USD path và default 64env;
[PPO config](source/g1_datn/g1_datn/tasks/manager_based/g1_multiskill/agents/ppo_mlp_cfg.py) kế thừa BasePPORunnerCfg;
[trainer](scripts/rsl_rl/train.py) áp CLI và migration RSL-RL5;
[registry](source/g1_datn/g1_datn/tasks/manager_based/g1_multiskill/__init__.py) trỏ [G1MultiSkillEnv](source/g1_datn/g1_datn/tasks/manager_based/g1_multiskill/g1_multiskill_env.py).

Snapshot thật Stage5 smoke64, **chỉ đại diện run mới**:

| File | SHA-256 |
|---|---|
| [agent.yaml](logs/rsl_rl/g1_multiskill_baseline/2026-10-20_17-14-10/params/agent.yaml) | 0e3227386a940542f59e598d5b58c71519f8ad17160c857e4c4996cf621734ac |
| [env.yaml](logs/rsl_rl/g1_multiskill_baseline/2026-10-20_17-14-10/params/env.yaml) | d6bbe249b51e7b25c0dffcb42451f0d181b17e5502bdfbb97154ceb109683919 |

Source mặc định max_iterations 3000, env64, experiment g1_multiskill_baseline; Stage5 YAML có max_iterations 2. C1 CLI override 200/256 và experiment riêng. Không copy YAML Stage5 rồi tuyên bố đó là run 19/10 hoặc C1 đã chạy. Genuine C1 YAML chỉ xuất hiện khi future trainer resolve/dump cấu hình. Các local mdp placeholder và recurrent config không được đăng ký thay PPO MLP; không tự chuyển recurrent.

### 3.2 PPO/network đã xác minh

| Thông số | Giá trị thực tế/source-snapshot |
|---|---|
| Runner / algorithm | OnPolicyRunner / PPO |
| Actor observation | 480=96×5 history;3 angular +3 gravity +3 command +29 q +29 qdot +29 last_action |
| Critic observation | 495=99×5; thêm 3 linear velocity |
| History layout | Nối theo term,5 slots/term, flatten; không phải 5 full frames nối tùy ý |
| Actor MLP | 480→512→256→128→29, ELU |
| Critic MLP | 495→512→256→128→1, ELU |
| Normalization | actor/critic obs_normalization=false; Identity đã thấy runtime |
| Action dimension/distribution | 29/Gaussian, init_std 1,0, std_type=scalar; không đổi |
| Rollout steps/env/iteration | **24** |
| Learning rate / schedule | 0,001 / adaptive; LR thực không cố định ở 0,001 |
| Desired KL | 0,01 |
| PPO clip | 0,2 |
| Epochs / minibatches | 5 /4 |
| Optimizer / max grad norm | Adam /1,0 |
| Gamma / GAE lambda | 0,99 /0,95 |
| Entropy coefficient | 0,01 |
| Value loss coefficient | 1,0 |
| Clipped value loss | true |
| Normalize advantage per minibatch | false |
| RND / symmetry / shared CNN | null /null /false |
| clip_actions / check_for_nan | null /true |
| Resume | false trong source CLI default và Stage5 |
| Save interval |100; không đổi |
| Training seed đề xuất |42; Stage5 source/snapshot seed 42 |
| Episode-length init lúc learn | init_at_random_ep_len=true; không thay |
| TF32 / deterministic backend | TF32 enabled; cuDNN deterministic=false, benchmark=false; PhysX enhanced determinism=false |
| Source training cap |3000 mặc định;**C1 CLI 200** |
| C1 transitions/minibatch |6144/iteration;1536/minibatch nếu chia4;200×6144=1.228.800 |
| Thời gian mô phỏng rollout |24×0,02=0,48s/env/iteration;96s/env toàn 200 iterations; không phải wall-clock |

Seed42 không bảo đảm bitwise reproducibility. C1 không bật thêm --deterministic vì làm đổi backend behavior; lưu RNG/backend settings và gọi deterministic inference theo mean-policy khi eval.

### 3.3 Simulation, actions, posture và PD

| Trường | Giá trị đã kiểm |
|---|---|
| Sim/backend | PhysxCfg/PhysxManager, cuda:0; solver_type 1, TGS |
| Physics dt / decimation / control dt |0,005s /4 /0,02s |
| Episode horizon / finite-horizon |20s /is_finite_horizon=false |
| Gravity |(0,0,−9,81)m/s² |
| Scene default/spacing |64env trong local source;2,5m; C1 override256 |
| Terrain |generator 9rows ×21cols, size 8×8m; chỉ subterrain flat MeshPlane, không rough dù tên COBBLESTONE |
| Terrain curriculum |true trong snapshot; không đổi trong C1 |
| Asset |official G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd |
| Action |JointPositionAction,29 joints, scale 0,25, use_default_offset=true, no action clip |
| Root initial pose |z0,8m, quaternion xyzw(0,0,0,1); lin/ang velocity 0 |
| Default joint velocity |mọi joint 0 |
| Default posture |hip pitch−0,1; knee 0,3; ankle pitch−0,2; shoulder pitch 0,3; shoulder roll L/R+0,25/−0,25; elbow 0,97; wrist roll L/R+0,15/−0,15rad; còn lại 0 |
| Soft joint limit factor |0,9; hard/velocity limits giữ asset |
| Contact sensor |history 3, track_air_time=true, update_period0,005s; không bật raw contact trong C1 |

q_target=q_default+0,25×action. C1 không nhân α hoặc bias targets. Tất cả actuator implicit; armature 0,01.

| Nhóm | Kp(Nm/rad) | Kd(Nm·s/rad) | Effort limit(Nm) | Velocity limit(rad/s) |
|---|---:|---:|---:|---:|
| Hip pitch/yaw |100|2|88|32|
| Hip roll |100|2|139|20|
| Knee |150|4|139|20|
| Waist yaw |200|5|88|32|
| Waist roll/pitch |40|5|25|37|
| Ankle pitch/roll |40|2|25|37|
| Shoulder/elbow/wrist roll |40|1|25|37|
| Wrist pitch/yaw |40|1|5|22|

### 3.4 Reward terms — giữ nguyên toàn bộ 19 weights

Nguồn env.yaml.rewards và Unitree RewardsCfg. Weights không là giá trị reward đã đo; reward manager/lifecycle có chuẩn hóa theo dt.

| Term | Weight | Ý nghĩa/params đáng chú ý |
| --- | --- | --- |
| track_lin_vel_xy | 1.0 | yaw-frame exponential, std 0,5 |
| track_ang_vel_z | 0.5 | exp, std 0,5 |
| alive | 0.15 | giữ function/params từ snapshot |
| base_linear_velocity | -2.0 | lin_vel_z_l2 |
| base_angular_velocity | -0.05 | ang_vel_xy_l2 |
| joint_vel | -0.001 | giữ function/params từ snapshot |
| joint_acc | -2.5e-07 | giữ function/params từ snapshot |
| action_rate | -0.05 | giữ function/params từ snapshot |
| dof_pos_limits | -5.0 | giữ function/params từ snapshot |
| energy | -2.0e-05 | Unitree energy |
| joint_deviation_arms | -0.1 | shoulder/elbow/wrist |
| joint_deviation_waists | -1 | waist.* |
| joint_deviation_legs | -1.0 | hip roll/yaw |
| flat_orientation_l2 | -5.0 | giữ function/params từ snapshot |
| base_height | -10 | target root height 0,78m |
| gait | 0.5 | period 0,8s, offset=[0.0, 0.5], threshold 0,55; gate command norm>0,1 |
| feet_slide | -0.2 | giữ function/params từ snapshot |
| feet_clearance | 1.0 | std 0,05, tanh_mult 2, target foot world-z 0,1m |
| undesired_contacts | -1 | threshold 1N, non-ankle bodies |

Không thay clearance/gait/survival penalties trong C1. Gait source gate command norm>0,1; clearance có speed weighting nên không kết luận nó ép nhấc chân lúc đứng chỉ từ weight.

### 3.5 Commands, curriculum, events, observations và termination

Commands base_velocity: resample10s, heading_command=false, rel_standing_envs0,02, rel_heading_envs1,0. Initial vx/vy/wz ranges đều[−0,1;0,1]. Limit ranges vx[−0,5;1,0], vy[−0,3;0,3], wz[−0,2;0,2]; đó không phải dải đã học. Built-in success thresholds0,5m/s và0,4rad/s không thay survival gate.

Curriculum gồm terrain_levels_vel và lin_vel_cmd_levels. Hàm sau xét reward track_lin_vel_xy>0,8×weight tại bước chia hết episode limit, rồi nới x/y range mỗi phía0,1 trong limit_ranges; angular curriculum không được đăng ký. Vì curriculum còn hoạt động, range thực trong train/eval có thể thay theo rollout. Snapshot trainer dump **trước learn**, không phải snapshot runtime cuối curriculum.

| Event | Mode | Params phải giữ |
|---|---|---|
| physics_material |startup|mọi bodies; μs/μd uniform0,3…1,0; restitution0;64buckets|
| add_base_mass |startup|torso_link add[−1,+3]kg; inertia theo implementation đã pin|
| base_external_force_torque |reset|torso force/torque ranges0|
| reset_base |reset|x/y±0,5m, yaw±3,14rad; mọi root velocities0|
| reset_robot_joints |reset|position scale[1,1]; velocity scale[−1,1]×default0→qdot0|
| push_robot |interval|mỗi5s, per-env timer; set world vx/vy±0,5m/s; không tắt trong C1/eval chuẩn|

Actor noise enabled: angular uniform±0,2 trước scale0,2→±0,04; gravity±0,05; qrel±0,01; qdotrel±1,5 trước scale0,05→±0,075. Critic corruption=false. Giữhistory5 và tất cả noise/modifiers.

Termination: time_out20s có time_out=true; root height<0,2m; bad_orientation>0,8rad. Giữ ngưỡng và phân biệt physical failure với time limit khi phân tích.

### 3.6 Trạng thái phần mềm và giới hạn audit

Python3.12.13, Sim6.0.1.0, Torch2.10.0+cu128, RSL-RL5.0.1; driver hiện595.91.07. Có6metadata conflicts đã biết từ Stage4B/5, chưa sửa; smoke/reset PASS không chứng minh long-run compatibility. Không uv sync từ root lockfile cũ. OMP_NUM_THREADS mặc định4; không thay LD_PRELOAD/VirtualGL.

Source và YAML được kiểm tĩnh, không import cấu hình để khởi động Kit. Resolved C1 runtime config, memory256env, future losses/weights/gradients đều **UNVERIFIED UNTIL EXECUTION**. Nếu future genuine C1 config khác whitelist CLI overrides, dừng audit, không tự chữa.

## 4. Historical reproducibility limitations

C0 là model_199.pt, run2026-10-19_08-46-28; SHA-256 **418fd00f1c20c8d6ada52746943471cb66e50f2ad5afc354d8bff5ee35a578d6**. Chỉ dùng archived evaluations làm ngữ cảnh; không resume.

**C0 EVALUATION REPRODUCIBILITY — BLOCKED.** Thiếu agent.yaml, env.yaml và smoke_verification.json gốc, 18 raw directories Mốc B cùng contact-onset provenance. Không chạy C0 với YAML Stage5 rồi tuyên bố exact recovery.

| Điều kiện so sánh | Bằng chứng và trạng thái |
|---|---|
| Architecture, actions, dt và PPO24 | Khớp mô tả historical docs/current source; không chứng minh exact snapshot cũ |
| Checkpoint binary/hash | Có và được bảo toàn |
| Reward, PD, noise, randomization, runtime curriculum lịch sử | Nhiều giá trị giống theo docs; exact equivalence **UNVERIFIED** |
| Asset/backend versions | Pin và docs hỗ trợ tương đồng; machine/driver khác, provenance thiếu |
| Initial weights, RNG và trajectories | Không có bằng chứng khớp; không phải bitwise replay |
| Archived C0 so C1 | Được mô tả như tiến bộ theo ngữ cảnh; không causal A/B chỉ khác reset |
| C1 model0/100/199/zero trên protocol mới | Có thể kiểm comparability bằng snapshots và initial-state fingerprints; chưa mặc định paired |

Nếu cần controlled retraining, thiết kế riêng **C1-R**: cùng current config, initial actor/critic weights và seeds; một nhánh reset fix, một nhánh test-only bypass. Ba training seeds 42/43/44 ×hai nhánh256×200 =**7.372.800 transitions**, sáu training runs. Phải duyệt chi phí và việc train trên known-bug pipeline riêng. Không cần thực hiện C1-R để cho phép C1; không ngầm phê duyệt nó.

## 5. Experiment C0/C1 design

**C0:** reference từ archived data; không execution. Chi phí training/evaluation mới của C0 trong kế hoạch chính bằng0; recovery dữ liệu là việc riêng.

**C1 hypothesis:** fresh PPO trên pipeline reset-corrected có thể cải thiện survival so early-policy baseline dưới cùng protocol. Không thêm PD/reward/posture intervention. Outcome hợp lệ gồm pipeline PASS nhưng insufficient improvement hoặc standing FAIL. Không chọn kết luận theo reward curve.

C1:256 env,200 iterations, seed 42, fresh initialization, no resume, cuda:0, TensorBoard. Experiment **g1_multiskill_resetfix_pilot**, run suffix **resetfix_seed 42_200**. Actual output là timestamp_resetfix_seed 42_200 dưới logs/rsl_rl/g1_multiskill_resetfix_pilot. Directory chưa được tạo; future agent phải ghi path thật từ native log, không đoán timestamp.

Whitelist overrides: num_envs64→256; source max_iterations 3000→200; experiment/run name, logger/output paths. Seed42 và cuda:0 giữ như reference mới; visualizer none, kit_first, verify_articulation dùng runtime đã kiểm. Không thêm Hydra override hoặc resume/checkpoint/load_run.

Giữ save_interval100. Source lưu **model_0 sau iteration đầu**, **model_100 sau101 iterations**, và **model_199 sau200 iterations nhờ final save**. Không yêu cầu model_99/model_200 vì source không lưu chúng. Model0 không phải policy chưa train. Giữ cả ba; không đổi save frequency. Crash có thể mất phần tiến trình giữa hai lần lưu; không tự restart/resume.

Candidate chính được định trước là model199; model100 là diagnostic trung gian; model0 là early-policy baseline. C1 có một training seed; ba eval seeds không trở thành ba training replicates.

Execution cần backup/cost gates, unique paths, resource checks và evaluation coverage được duyệt. Dừng ở200; không kéo lên3000. Benchmark riêng cần approval riêng, không ngầm tạo pilot thứ hai.

## 6. Future C2/C3/C4 ablations — độc lập C1

| Experiment | Một biến can thiệp | Các biến giữ nguyên | Metric/điều kiện |
|---|---|---|---|
| C2a: settling | Current spawn so một protocol settle định nghĩa trước | Cùng fixed policy, seeds, PD, reward, commands, termination, horizon 20s | Contact readiness, qdot, root/COM velocity, impact, first survival; tách settling clock và evaluation clock |
| C2b: posture | Một ankle target bias+0,10rad | Cùng policy/gains/commands/seeds; không đồng thời đổi settling | Tracking, COM drift, survival, contacts; ghi nhãn target intervention |
| C3: ankle PD | Kp pitch ×1/×2/×4/×8; giữ Kd2 | Cùng policy,3eval seeds, initial physical distribution, commands, termination, horizon | Joint errors, effort estimates/clipping, whole-body stability; không chọn chỉ theo ankle tracking |
| C4: standing curriculum | Chỉ command distribution→zero trong nhánh đầu | Reset fix, PD, posture, reward, PPO giữ nguyên; fresh seed/init theo thiết kế | So C1/reference bằng evaluation độc lập; không gộp reward changes |
| C4: reward sau đó | Một term/weight hoặc gating | Standing protocol đã duyệt và các biến khác giữ nguyên | Log raw/weighted terms; survival gate trước mean reward |

C2a readiness đề xuất: hai chân có contact report hợp lệ, root linear norm<0,05m/s, angular norm<0,1rad/s và qdot RMS<0,1rad/s liên tục0,2s; settling cap1s. Đây là ngưỡng kỹ thuật đề xuất, chưa tối ưu đã chứng minh. Báo số robot settle/fail; không loại robots chưa settle để làm đẹp denominator.

C3:4arms×3seeds×64env×20s =**768.000 transitions/scenario**. C2 với hai arms =**384.000**. Factorial settling×gain cần thiết kế/preregister riêng. Chọn một fixed policy đã có của C1; zero-action reference là arm riêng. Không thay policy theo từng gain.

C4 chỉ xem xét sau phân tích C1 hoặc bằng chứng C2/C3 đủ để chọn control baseline mới. GateB FAIL cho phép **đề xuất** C2/C3/C4 để duyệt, không tự thay config. GateC chưa PASS thì chưa chuyển sang walking curriculum khó hơn.

## 7. Evaluation protocol

### 7.1 Ma trận đề xuất

64env, horizon 20s, eval startup/reset seeds **42,43,44**. Mean deterministic inference; giữ noise, randomization, pushes và termination. Scenarios: standing(0,0,0), forward_slow(vx=0,1m/s), training_commands. Zero chỉ standing, dùng genuine config C1 cuối.

Mỗi seed/scenario là một invocation riêng, để **actual reset seeds đúng42/43/44 cho từng scenario**. Khi nhiều scenarios trong một process, source reset với args.seed+scenario_index; không được ghi mọi scenario có cùng seed 42.

Ba checkpoints ×ba scenarios +một zero arm =10conditions/seed; tổng **30conditions**, mỗi condition64×1000steps: **1.920.000 transitions**. Chín Kit invocations, ba hoặc bốn scene rollouts mỗi invocation. Không concurrent simulator; không đưa C0 thiếu snapshots vào checkpoint list.

Zero và model0/100/199 dùng cùng genuine C1 config. Comparability phải kiểm bằng initial fingerprint: relative root pose, root velocity, q/qdot, mass/inertia/material, gains/limits, observations và commands. Cùng startup/reset seed chưa đủ để tự nhận paired equivalence. Policy weights khác là treatment; không đòi trajectories sau action giống nhau.

Training_commands bắt đầu từ saved initial config, curriculum còn hoạt động. Không mặc định đây là final training distribution hoặc cùng command sequence giữa policies. Log actual command/range và curriculum state; scenario này là supplementary evaluation, không phải causal paired tracking test.

### 7.2 CLI và coverage thực tế

[evaluate_multiskill.py](scripts/rsl_rl/evaluate_multiskill.py) hỗ trợ --checkpoints,--output,--num_envs,--seconds,**--seed**,--scenarios,--zero_reference. AppLauncher hỗ trợ --device/--visualizer. Không có --seeds,--kit_first hoặc --task trong evaluator; script tự bootstrap AppLauncher trước config imports.

Có10 feature channels: vx/vy/vz theo body frame, wz, tilt, relative root height, omegaXY norm, action RMS, joint-velocity RMS, joint deviation từ default. Có commands, terminal flags, episode steps, first-done index và dt. Actor weights được so với checkpoint; features/actions/rewards được kiểm finite. Không optimizer update trong evaluation.

Chưa có roll/pitch riêng, whole-body COM, full actions/q/qdot/targets/limits/torque/contact forces hoặc initial pairing fingerprints. **Joint deviation từ default không phải target-tracking error.** Không gọi aggregate joint-velocity RMS là kiểm đủ từng joint.

Các bẫy phải xử lý offline:
1. first_failure_steps bắt mọi done, kể cả timeout. Trường survival_horizon=(first<0) có thể bằng0 ngay khi robot timeout đúng20s. Không dùng trực tiếp làm GateC.
2. Trường survival2/5 hiện dùng >=threshold; kế hoạch dưới chọn conservative tie rule, phải recompute nhất quán.
3. censored_episodes ban đầu hardcode num_envs; completed-episode median không phải first-failure median.

[summarizer hiện tại](scripts/rsl_rl/summarize_multiskill_eval.py) hardcode **13conditions**, hai model ×sáu scenarios+zero. --input hợp lệ nhưng data contract không phù hợp C1. Không chạy nó trên ma trận mới; dùng offline proposal ở mục13.

### 7.3 Change proposal cho full evaluation metrics — CHƯA APPLY

Đề xuất test-level instrumentation cho evaluator; giữ CLI và10features hiện tại, thêm NPZ arrays riêng. Không đổi production config, trainer, reward hoặc dependency. Cần future approval/review trước evaluation chính. Diff chỉ mô tả vị trí/contract, chưa là implementation đã test:
```diff
--- scripts/rsl_rl/evaluate_multiskill.py (current)
+++ scripts/rsl_rl/evaluate_multiskill.py (proposed test instrumentation)
@@ after reset, before rollout
+ # Save initial fingerprints: relative root pose/vel, q/qd, observations,
+ # masses/inertias/materials, Kp/Kd/limits, commands and env origins.
+ # Give t=0 contact records explicit validity/timestamp flags.
@@ in measured_compute, before auto-reset
+ # Record root quaternion -> roll/pitch and mass-weighted COM/velocity.
+ # Record full actions/q/qd/targets, computed/applied torque estimates,
+ # effort limits, left/right forces and contact timing.
+ # Preserve the existing features/commands/termination arrays.
@@ save NPZ and manifest
+ # Save additional arrays with schema/units/frames/timestamps.
+ # Record env_class, source hashes, command/curriculum ranges and pairing status.
```

Review timing/frames/units và static syntax, rồi bounded runtime verification trong bước được duyệt. Archive evaluator diff/hash cùng run. Nếu extension chưa được phê duyệt, core evaluation vẫn cho survival/tilt/root height/RMSE, nhưng COM/roll-pitch/action-rate/joint/torque diagnostics và GateC stability audit còn **UNVERIFIED**. Không công bố full locomotion PASS dựa trên metrics chưa có.

## 8. Metrics và statistical interpretation

Physical failure là bad_orientation, base_height hoặc physical termination khác. **time_out là administrative censoring**, không phải ngã. First trajectory bắt đầu reset, chứa terminal sample và kết thúc trước auto-reset. Nếu physical flag và timeout cùng bước, ưu tiên physical failure.

S(t) là tỷ lệ không có physical failure tới và gồm t. Physical failure đúng threshold tính FAIL; timeout-only đúng20s tính đã sống đủ20s. Tái tính S2/S5/S10/S20 từ NPZ flags. Median/p10 là quantiles first-physical-failure từ empirical survival/KM; nếu chưa chạm trong20s, ghi **NOT REACHED (≥20s)**, không thay censor bằng event20s hoặc extrapolate. Tỷ lệ median2× có thể không xác định nếu cả hai arms đều censor ở20s.

| Metric | Cách đo và coverage |
|---|---|
| First survival2/5/10/20 | Recompute terminal flags; không dùng raw horizon summary |
| Median/p10 | KM/empirical survival với censor20s; per-seed và pooled |
| Bad-orientation rate | First failures và all completed episodes tách denominator; timeout riêng |
| Tilt p95 | First trajectory only, acos(−gravity_z); không trộn sau reset |
| Roll/pitch distributions | Cần quaternion/RPY extension; đơn vị rõ |
| Root height | Relative env origin, m; không thay bằng COM height |
| COM drift | Mass-weighted COM−initial COM; ghi world/initial-yaw frame; cần extension |
| XY RMSE | sqrt(mean((vx−cmdx)²+(vy−cmdy)²)), m/s, body frame; first/pooled riêng |
| Yaw RMSE | sqrt(mean((wz−cmdwz)²)), rad/s; luôn kèm duration/exposure |
| Actions | Full RMS và RMS delta-action/controlstep; không nối qua reset; rate cần full actions |
| Joint tracking/velocity | qtarget−q RMS/peak đủ29 joints; qdot RMS/max; không dùng default deviation thay thế |
| Torque/clipping | Unclipped PD estimates so limits+1e−4; không phải motor-torque sensor |
| Foot timing/asymmetry | Valid load contact; abs(FzL−FzR)/(FzL+FzR) khi denominator>epsilon; air/invalid riêng |
| Support margin | Secondary cho đến khi C-T PASS; không fill NaN |

Seed/scene là cluster; không coi192robots hoặc nhiều resets/timesteps là IID. Báo ba seed riêng và pooled; ba eval seeds không thay training replicates. Paired changes chỉ dùng khi fingerprint PASS. Với ba clusters, CI/bootstrap rất yếu; không dùng p-value để claim generalization. Failure-time resolution là20ms.

Tilt p95 first-episode được weighted theo observed samples; kèm per-env p95 và worst-env summaries để không che failure ngắn. GateC xét từng seed, không chỉ pooled. RMSE thấp trong0,2s không là tracking success; phải kèm S20, exposure và termination rate.

## 9. Contact telemetry quality gate — C-T riêng

C-T không nhất thiết chặn pilot dùng survival/termination, nhưng **bắt buộc trước khi dùng support margin làm acceptance metric chính**. Đề xuất8env, seeds 42/43/44, zero reference, full/subset resets và nonzero pose/velocity probes; tối đa4s/seed, ít nhất hai full resets và bốn subset resets/seed. Có thể dùng genuine Stage5 checkpoint cho policy probe, không gọi là historical model199.

Ghi simulation/control/substep timestamps, reset masks, root/COM/foot poses, native velocities, raw points/normals/forces/separations/counts/filter paths, sensor net force, hull vertices/area/margin, CoP numerator/denominator và availability flags. Lấy mẫu trước reset, sau reset/forward và sau physics bước đầu. Không mặc định report t=0 có timestamp mới.

| Trạng thái | Điều kiện |
|---|---|
| No contact | Fresh valid measurement,0loaded eligible points/force; không suy từ NaN |
| <3usable points | Có load nhưng dưới3distinct XY points |
| Degenerate hull | Ít nhất3points nhưng collinear/area≤epsilon_area |
| COM outside valid hull | Fresh, valid hull, area>epsilon, margin<−epsilon_distance |
| CoP available | Fresh valid forces, positive denominator; độc lập việc hull có3đỉnh |
| CoP unavailable | Không usable load/denominator0; ghi lý do |
| Invalid/missing | Timestamp stale, unexpected nonfinite, missing/inconsistent buffer; không gộp với air |

Ngưỡng đề xuất: force1e−4N, abs(normal_z)>0,5, uniqueXY tolerance1e−6m, epsilon_area1e−8m², epsilon_distance1e−5m. Normal-force CoP và net world-Fz có thể dùng tập khác nhau. Validate normal orientation/units trước weighted CoP; không gọi normal CoP là full ZMP.

Acceptance C-T: mọi reset probe có timestamp/mask phân loại được; **0stale samples bị gắn VALID**; offline hull recomputation khớp valid margin/vertices tới1e−5m; CoP nằm trong convex hull của cùng eligible point set tới1e−5m. Report đầy đủ no-load, degenerate, invalid counts, không drop NaN. Kiểm đồng bộ sensor/raw-force conventions; chưa đặt force-consistency bound khi conventions chưa rõ. Một lỗi timestamp hệ thống chưa giải thích làm gate FAIL.

GPU/time/VRAM C-T **UNKNOWN**. Cần approval riêng cho implementation, dataset, wall-time cap, cost và backup trước khi chạy. Không chạy lại18MốcB experiments ở Stage8.

## 10. PPO acceptance gates

| Gate | Tiêu chí đề xuất | Không đồng nghĩa |
|---|---|---|
| A: Training pipeline | Đúng29DoF/actions, class, PhysX CUDA, reset fix; finite losses/weights/optimizer buffers; actor/critic updates; không fatal CUDA/PhysX; đủ checkpoint/config/log/hash; iterations0…199 và1.228.800transitions | Locomotion PASS |
| B: Pilot improvement | S5≥50% và median survival≥2×baseline có comparability hợp lệ | Standing20s hoặc causal reset A/B |
| C: Standing readiness | Mỗi seed≥90%first episodes sống20s, ít nhất58/64; first-episode tilt p95<20°; ít nhất3eval seeds; không instability nghiêm trọng ở joint/torque | Walking, running, jumping hoặc rough-terrain PASS |

GateB baseline chính là **C1 model_0** evaluated cùng config/source/protocol, noise và initial fingerprints. Đây là tiến bộ trong một training run, không kiểm riêng causal reset effect. Report thêm zero reference. Không dùng archived C0 khác protocol để claim2×. Nếu pairing FAIL hoặc median ratio không xác định vì censoring, ghi **COMPARABILITY UNVERIFIED / INCONCLUSIVE**; S5 đánh giá riêng.

Khi median C1 chưa reached nhưng lower bound20s đã ≥2×median baseline xác định, có thể chứng minh ngưỡng2× bằng lower bound và phải ghi rõ. Nếu cả hai medians chưa reached, không suy ra ratio2×.

GateC dùng first trajectories theo từng seed; không cộng nhiều reset samples upright. Joint/torque audit đề xuất: không NaN, không hard-bound excess>1e−3rad kéo dài, không velocity-limit exceedance chưa giải thích. PD clipping không tự động là motor failure; report joint, severity, duration. TL phải duyệt numerical clipping threshold nếu muốn dùng nó làm gate. Thiếu telemetry thì stability audit UNVERIFIED, không full GateC.

GateA phải đọc actual return code lẫn artifacts. Trainer hiện nuốt KeyboardInterrupt; exit0 đơn lẻ chưa đủ. Kiểm model0/100/199, iter fields, genuine snapshots, smoke_verification200iterations/4800controlsteps, TensorBoard finite scalars và weight differences. check_for_nan kiểm chủ yếu obs/rewards/dones, không chứng minh mọi gradient/advantage finite; cần additional checks được review nếu muốn claim rộng hơn.

GateA FAIL: preserve logs, dừng, không tự chạy C2/C3/C4. GateB FAIL: **INSUFFICIENT IMPROVEMENT**, stop200, phân tích; không tăng iterations/đổi reward. GateB PASS nhưng GateC FAIL: đề xuất C2/C3 hoặc standing curriculum để duyệt. GateC PASS mới request walking curriculum khó hơn. Mọi ngưỡng vẫn là proposal, chưa là kết quả đạt hoặc tiêu chí execution được phê duyệt.

## 11. Compute/time/cost planning

RTX3090 có24576MiB, driver595.91.07; CPU quota15,36, RAM limit60,31GiB, disk200GiB. Free disk khoảng168,72GiB tại audit, filesystem overlay. Chỉ một Kit process tại một thời điểm.

| Bằng chứng/ước lượng | Giá trị | Giới hạn |
|---|---|---|
| Stage5 wall-clock | 1env36,01s;64env25,98s | Chỉ2iterations, cold/warm khác; không extrapolate tuyến tính |
| Smoke64 sampled GPU/RAM | 3136MiB≈3,06GiB; container11,39GiB | Whole GPU/container mỗi2s, không peak isolated |
| 256env VRAM/RAM | **UNKNOWN** | Physics/terrain buffers, cache và rollout không scale đơn giản |
| Obs rollout buffers | 6144×(480+495)×4=23.961.600bytes≈22,85MiB | Chỉ float32 observations, không total VRAM |
| Ba checkpoints | 3×10.010.839=30.032.517bytes≈28,64MiB | Theo same-architecture smoke; future size phải kiểm |
| Core eval raw | Khoảng120,96MB uncompressed+headers cho30conditions | 10f32features,3f32commands, int64lengths,3boolterms; chưa extended telemetry |
| Extended telemetry | **UNKNOWN**, reserve≥5GiB riêng | Full joints/contacts tăng dữ liệu; không video |
| Working disk reserve | ≥20GiB free, đề xuất≥5GiB train/provenance và≥5GiB eval | Operational budget, không dự báo chính xác |
| Train/eval/C-T wall time | **UNKNOWN** | Chưa benchmark256env hoặc20s evaluation protocol mới |
| Current hourly rate/USD | **UNKNOWN** | Không đoán giá và không query giá trong Stage8 offline |

Cost=p×billable hours, với p là verified instance hourly price. Tách C_train=p·h_train; C_eval=p·h_eval; C_startup_diagnostic=p·h_startup_CT; C_idle_backup=p·h_idle_backup. Total là tổng các phần; storage/network có giá verified riêng. Nếu wall train/eval đã gồm startup, phải trừ startup trước khi cộng để tránh đếm đôi. Chưa có p hoặc approved wall/cost cap nên không đưa số USD.

Benchmark tương lai nếu được duyệt:256env×20iterations ở run riêng, cùng C1 config ngoài iteration cap; ghi startup, collection/learn/save time từng iteration và resource samples1–2s. Chi phí riêng122.880transitions và một startup. Không resume C1 từ benchmark. Bỏ cold iterations khi mô tả steady median/p10/p90; dự báo T_start+200·t_iter+I/O phải kèm range/assumptions, không gọi là measurement.

Phương án ít tốn hơn: đo10–20iterations đầu trong C1 đã được duyệt, không thêm run, không đổi cap200. Vẫn phải chốt abort budget trước khi start.

Resource guards đề xuất: trước start free GPU≥8GiB, free RAM≥20GiB, free disk≥20GiB. Monitor VRAM/cgroup memory/disk; dừng nếu RAM>90%limit, free disk<10GiB hoặc fatal OOM. Gần90%VRAM cần review trước cấp phát lớn. Đây là budget đề xuất, không chứng minh256env fit; không tự giảm env rồi claim cùng C1.

Giảm chi phí: headless/không video, một simulator, ba checkpoints và scenarios định trước; gộp checkpoints trong một process mỗi seed/scenario; offline analysis, nén lossless, incremental backup. Không xóa dữ liệu lịch sử, nới acceptance hoặc rút horizon 20s để tiết kiệm rồi claim GateC.

## 12. Backup and persistence plan

**TRAINING BACKUP GATE — NOT SATISFIED.** Chưa có destination độc lập được phê duyệt. /workspace và /tmp, hai folders cùng instance hoặc một volume chưa xác minh không phải independent backup. Không tự upload hoặc push.

Destination đề xuất: object-storage bucket hoặc SFTP server thuộc tài khoản người dùng, khác instance. URI/quyền truy cập chờ duyệt. Namespace minh họa **<approved-destination>/g1-multiskill/instance56912565/C1/<run-id>/** không phải địa chỉ đã có.

Phải bảo vệ: genuine env/agent YAML; model0/100/199; TensorBoard; stdout/stderr/exit code; run IDs/seeds/command; source HEAD+tracked hashes+local diff; plan/MốcB report; asset manifest/4USD hashes; Lab/Unitree commits+patch; package inventories; evaluation raw/JSON/offline-code/schema/metrics; contact quality data nếu có. Không dump toàn environment hoặc lưu tokens/passwords.

Lịch backup: pre-run source/config archive và restore test; genuine YAML/model0 ngay khi hình thành; model100 ngay sau khi lưu; final199/logs/evaluation khi hoàn tất. Chỉ copy checkpoints đã đóng hoặc size/hash ổn định; TensorBoard đang ghi phải đánh nhãn partial hoặc đợi flush. Không overwrite external versions cũ.

Mỗi object có bytes và SHA-256 local; copy rồi đọc/download independent remote copy để rehash và so exact. ETag không mặc định SHA-256. Chỉ backup PASS khi remote bytes/hash thực sự khớp; ghi timestamp/error. Backup failure phải dừng để review theo approved execution policy, không xóa local hoặc tự resume sau crash.

Restore: máy mới với approved pinned stack và asset manifest; source HEAD+local diff; verify hashes; đặt checkpoint cùng đúng YAML của run; read-only audit trước. Simulation regression/evaluation/resume vẫn cần approval riêng. Không uv sync hoặc upgrade drivers để thử sửa.

Không được xóa model100/199 lịch sử,14MốcB artifacts, Stage5/6 logs/checkpoints/snapshots, MOC_B_FINAL_REPORT, plan MốcC, C1 cả ba checkpoints/genuine configs/eval raw/backup manifests. Giữ toàn bộ pilot tới TL review; retention/pruning sau đó là quyết định riêng.

## 13. Commands — DOCUMENT ONLY, NOT EXECUTED

Các blocks được kiểm tĩnh bằng CLI source/AST và shell/Python syntax, khôngimportproject hoặcchạycommand. Runtimepaths vàapprovedwallcap chưa có; futureagent resolve từgenuineoutput, không yêu cầungười dùng tựgõlệnh. **Chỉ thực hiện sau execution approval riêng.**

### 13.1 C1 training proposal — NOT EXECUTED

Không có cờ --no-resume; default --resume=false được giữ bằng cách **không truyền** --resume, --checkpoint hoặc --load_run. Không dùng shortcut --smoke64. Các biến budget phải được future agent điền từ approval, không lấy một giá tự đoán.

```bash
set -euo pipefail
cd /workspace/Humaniod-MultiSkill
test "$(git rev-parse HEAD)" = fc46f39ef0214e84cb0bb484dfec0d1fef7ba161
test "$(git branch --show-current)" = g1-multiskill
test ! -e logs/rsl_rl/g1_multiskill_resetfix_pilot
test -n "$TRAIN_WALL_CAP"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
audit_dir="run/moc-c-C1-$stamp"
test ! -e "$audit_dir"
mkdir "$audit_dir"
timeout --signal=TERM --kill-after=30s "$TRAIN_WALL_CAP" \
  env OMNI_KIT_ACCEPT_EULA=YES \
  ./run_multiskill.sh scripts/rsl_rl/train.py \
    --task G1-MultiSkill-v0 \
    --num_envs 256 --max_iterations 200 --seed 42 \
    --experiment_name g1_multiskill_resetfix_pilot \
    --run_name resetfix_seed42_200 \
    --logger tensorboard --device cuda:0 \
    --visualizer none --kit_first --verify_articulation \
    2>&1 | tee "$audit_dir/train.log"
```

TRAIN_WALL_CAP là approved timeout string, chưa có giá trị. Đây là guard thêm ngoài iteration cap200, không phải ước lượng runtime. Nếu experiment root đã có do lần chạy khác, dừng; không xóa/ghi đè để chạy tiếp. Mọi abort giữ log/partial artifacts; không tự retry/resume.

Ghi actual timestamped run path từ native log vào C1_RUN_DIR. Cần backup pre-run provenance và genuine YAML/model0 ngay khi có, theo mục12. Command trên chưa bao gồm external upload do destination chưa được phê duyệt.

### 13.2 Checkpoint/config audit proposal — NOT EXECUTED

C1_RUN_DIR do future agent resolve từ run thật, không phải path Stage5. Kiểm file tồn tại/nonempty, checkpoint binary, iter đúng và tensor finite; snapshot resume=false phải được kiểm bằng loader snapshot sau khi runner đã dump thật. Exit0 của trainer không đủ.

```bash
set -euo pipefail
cd /workspace/Humaniod-MultiSkill
test -n "$C1_RUN_DIR"
test -s "$C1_RUN_DIR/params/agent.yaml"
test -s "$C1_RUN_DIR/params/env.yaml"
test -s "$C1_RUN_DIR/smoke_verification.json"
sha256sum "$C1_RUN_DIR"/model_0.pt \
          "$C1_RUN_DIR"/model_100.pt \
          "$C1_RUN_DIR"/model_199.pt \
          "$C1_RUN_DIR"/params/agent.yaml \
          "$C1_RUN_DIR"/params/env.yaml
env C1_RUN_DIR="$C1_RUN_DIR" PYTHONDONTWRITEBYTECODE=1 \
.venv-multiskill/bin/python -B - <<'PY'
import json, os
from pathlib import Path
import torch
run = Path(os.environ["C1_RUN_DIR"])
checkpoints = {}
def finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value: finite(item)
for iteration in [0, 100, 199]:
    path = run / f"model_{iteration}.pt"
    assert path.stat().st_size > 1024
    assert not path.read_bytes().startswith(b"version https://git-lfs")
    data = torch.load(path, map_location="cpu", weights_only=True)
    assert data["iter"] == iteration
    for key in ["actor_state_dict", "critic_state_dict", "optimizer_state_dict"]:
        finite(data[key])
    checkpoints[iteration] = data
for group in ["actor_state_dict", "critic_state_dict"]:
    assert any(not torch.equal(v, checkpoints[199][group][k])
               for k, v in checkpoints[0][group].items())
evidence = json.loads((run / "smoke_verification.json").read_text())
for key, value in {"num_envs": 256, "physics": "PhysxCfg",
                   "device": "cuda:0", "num_dof": 29,
                   "num_controlled_joints": 29, "action_dim": 29,
                   "ppo_iterations": 200}.items():
    assert evidence[key] == value
assert evidence["physics_control_steps"] == 4800
print("Checkpoint/articulation audit PASS; inspect YAML/TensorBoard separately")
PY
```

Không cần unpickle fallback nếu weights_only thất bại. Actor/critic finite và khác weights không chứng minh mọi gradient/advantage từng bước finite. Optimizer state fields/step counters, loss scalars/KL/LR và app return code còn phải kiểm trong full GateA audit.

### 13.3 Evaluation commands đã khớp CLI — NOT EXECUTED

Dùng chính genuine C1 snapshots cùng run. Ba scenarios chạy trong **các invocations riêng** để không bị seed offset. Thêm zero_reference chỉ cho standing. Mỗi output cần directory mới vì evaluator dùng exist_ok=True và có thể overwrite.

```bash
set -euo pipefail
cd /workspace/Humaniod-MultiSkill
test -n "$C1_RUN_DIR"
test -n "$EVAL_WALL_CAP"
for checkpoint in model_0.pt model_100.pt model_199.pt; do
  test -s "$C1_RUN_DIR/$checkpoint"
done
stamp=$(date -u +%Y%m%dT%H%M%SZ)
C1_EVAL_DIR="run/moc-c-C1-eval-$stamp"
test ! -e "$C1_EVAL_DIR"
mkdir "$C1_EVAL_DIR"
export C1_EVAL_DIR
for eval_seed in 42 43 44; do
  for scenario in standing forward_slow training_commands; do
    output="$C1_EVAL_DIR/seed$eval_seed-$scenario"
    test ! -e "$output"
    extra=()
    if [[ "$scenario" == standing ]]; then
      extra+=(--zero_reference)
    fi
    timeout --signal=TERM --kill-after=30s "$EVAL_WALL_CAP" \
      env OMNI_KIT_ACCEPT_EULA=YES \
      ./run_multiskill.sh scripts/rsl_rl/evaluate_multiskill.py \
        --checkpoints "$C1_RUN_DIR/model_0.pt" \
                      "$C1_RUN_DIR/model_100.pt" \
                      "$C1_RUN_DIR/model_199.pt" \
        --num_envs 64 --seconds 20 --seed "$eval_seed" \
        --scenarios "$scenario" --visualizer none --device cuda:0 \
        "${extra[@]}" \
        --output "$output" \
        2>&1 | tee "$C1_EVAL_DIR/seed$eval_seed-$scenario.log"
  done
done
```

Không truyền --seeds/--kit_first/--task cho evaluator vì parser hiện không hỗ trợ. List checkpoints chỉ trong **một run C1** để tránh hai checkpoint cùng stem ghi đè output. Evaluation extension mục7.3 phải được review/approve trước nếu cần full metrics; nó giữ nguyên các CLI này.

### 13.4 Offline first-episode summarization — NOT EXECUTED

Không gọi summarizer lịch sử vốn hardcode13conditions. Inline helper dưới đây đọc đúng NPZ core hiện có, giữ raw nguyên bản, tạo summary mới. Nó tách timeout/physical failure, sửa horizon survival và tính censored first-failure quantiles. Full roll/pitch/COM/joint/torque/contact analysis từ extension cần schema-aware helper bổ sung, không giả dữ liệu cho fields chưa có.

```bash
set -euo pipefail
cd /workspace/Humaniod-MultiSkill
test -n "$C1_EVAL_DIR"
env C1_EVAL_DIR="$C1_EVAL_DIR" PYTHONDONTWRITEBYTECODE=1 \
.venv-multiskill/bin/python -B - <<'PY'
import csv, json, math, os
from pathlib import Path
import numpy as np
root = Path(os.environ["C1_EVAL_DIR"])
out = root / "analysis_first_episodes"
out.mkdir(exist_ok=False)
rows, details = [], []
result_files = sorted(root.glob("seed*-*/results.json"))
assert len(result_files) == 9
for result_file in result_files:
    manifest = json.loads((result_file.parent / "manifest.json").read_text())
    seed = manifest["seed"]
    for entry in json.loads(result_file.read_text()):
        model, scenario = entry["checkpoint"], entry["scenario"]
        path = result_file.parent / model / f"{scenario}.npz"
        with np.load(path, allow_pickle=False) as raw:
            features, commands = raw["features"], raw["commands"]
            lengths, flags = raw["terminal_episode_steps"], raw["terminations"]
            names = raw["termination_names"].tolist()
            dt = float(raw["dt"])
        T, N, _ = features.shape
        horizon = T * dt
        assert N == 64 and abs(horizon - 20.0) < 1e-8
        assert np.isfinite(features).all() and np.isfinite(commands).all()
        assert "time_out" in names
        ends = np.full(N, T, dtype=int)
        physical = np.zeros(N, dtype=bool)
        why = [[] for _ in range(N)]
        for env_id in range(N):
            hits = np.flatnonzero(lengths[:, env_id] > 0)
            if len(hits):
                k = int(hits[0])
                ends[env_id] = k + 1
                why[env_id] = [name for i, name in enumerate(names)
                               if flags[k, env_id, i]]
                assert why[env_id]
                physical[env_id] = any(name != "time_out"
                                      for name in why[env_id])
                # A timeout before horizon would be unexpected for a fresh reset.
                if not physical[env_id]:
                    assert abs(ends[env_id] * dt - horizon) < 1e-8
        observed = ends * dt
        failure = np.where(physical, observed, np.inf)
        keep = np.arange(T)[:, None] < ends[None, :]
        event_times = np.sort(observed[physical])
        def quantile(p):
            rank = int(math.ceil(p * N))
            return (float(event_times[rank - 1]), None) if len(event_times) >= rank else (None, horizon)
        p10, p10_bound = quantile(.10)
        median, median_bound = quantile(.50)
        error = features[:, :, :2] - commands[:, :, :2]
        yaw_error = features[:, :, 3] - commands[:, :, 2]
        row = {
            "seed": seed, "model": model, "scenario": scenario,
            "first_episodes": N, "physical_failures": int(physical.sum()),
            "censored_first_episodes": int((~physical).sum()),
            "median_failure_s": median, "median_lower_bound_s": median_bound,
            "p10_failure_s": p10, "p10_lower_bound_s": p10_bound,
            "bad_orientation_first_fraction": sum("bad_orientation" in x for x in why) / N,
            "first_exposure_robot_s": float(observed.sum()),
            "tilt_first_p95_deg": float(np.rad2deg(np.quantile(features[:, :, 4][keep], .95))),
            "root_height_first_mean_m": float(features[:, :, 5][keep].mean()),
            "xy_first_rmse_mps": float(np.sqrt(np.mean(np.sum(error[keep] ** 2, axis=1)))),
            "yaw_first_rmse_radps": float(np.sqrt(np.mean(yaw_error[keep] ** 2))),
            "action_first_rms": float(np.sqrt(np.mean(features[:, :, 7][keep] ** 2))),
            "joint_velocity_first_rms_radps": float(np.sqrt(np.mean(features[:, :, 8][keep] ** 2))),
        }
        for threshold in [2, 5, 10, 20]:
            row[f"S{threshold}"] = float(np.mean(failure > threshold))
        rows.append(row)
        details.append({"seed": seed, "model": model, "scenario": scenario,
                        "observed_s": observed.tolist(), "physical_failure": physical.tolist(),
                        "first_termination_names": why, "source_npz": str(path)})
assert len(rows) == 30
assert len({(r["seed"], r["model"], r["scenario"]) for r in rows}) == 30
assert {r["seed"] for r in rows} == {42, 43, 44}
(out / "summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
(out / "first_episode_records.json").write_text(json.dumps(details, indent=2, allow_nan=False))
with (out / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
```

Helper chưa được chạy với C1 vì C1 chưa tồn tại. Future execution cần test offline fixtures cho no-failure, timeout at20, physical failure đúng20, mixed flags, failure at2/5 và censoring; không dùng nó như verifier độc lập trước review. Nó dùng NumPy đã có trong môi trường, không cài package mới.

### 13.5 Artifact integrity/provenance — NOT EXECUTED

Tạo manifests ngoài các raw files; không đổi dữ liệu để hợp verifier. Chỉ hash files hiện có, không dump secrets.

```bash
set -euo pipefail
cd /workspace/Humaniod-MultiSkill
test -n "$C1_RUN_DIR"
git rev-parse HEAD
git status --short --untracked-files=all
git diff --binary
git -C .deps/IsaacLab-multiskill rev-parse HEAD
git -C .deps/unitree_rl_lab-multiskill rev-parse HEAD
env C1_RUN_DIR="$C1_RUN_DIR" PYTHONDONTWRITEBYTECODE=1 \
.venv-multiskill/bin/python -B - <<'PY'
import hashlib, importlib.metadata as m, json, os
from pathlib import Path
run = Path(os.environ["C1_RUN_DIR"])
target = run / "integrity_manifest.json"
assert not target.exists()
files = {}
for p in sorted(run.rglob("*")):
    if p.is_file():
        files[str(p.relative_to(run))] = {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
target.write_text(json.dumps({
    "files": files,
    "packages": sorted([[d.metadata["Name"], d.version] for d in m.distributions()])
}, indent=2))
print(target)
PY
```

Chạy tương tự với evaluation root sau khi đóng log/flush. Copy source HEAD+working diff, toàn bộ source/config có ý nghĩa và plan/report vào provenance archive; Git archive riêng không chứa hai tài liệu untracked mới. Remote destination chưa phê duyệt nên không có upload command. Backup verification phải rehash **independent remote copy**, không coi local manifest là backup đã đạt.

## 14. Risks and failure conditions

Rủi ro: C1 không exact historical A/B, một training seed, GPU nondeterminism, curriculum thay runtime, noise vẫn bật, output collisions, missing C0 snapshots, contact-quality hypothesis, sáu metadata conflicts, VRAM/time chưa biết, overlay loss và backup/cost cap chưa approved.

Không start nếu backup gate chưa đạt, budget/rate-or-cap thiếu, HEAD/source khác baseline chưa giải thích, output collision, config ngoài whitelist, missing asset, concurrent GPU job hoặc resource guards chưa đạt. Khi chạy: fatal CUDA/PhysX/OOM, nonfinite loss/weights, corrupt checkpoint, watchdog quá approved cap, backup failure hoặc disk guard vi phạm thì TERM theo policy đã duyệt và preserve outputs. Không auto retry/resume/đổi PD.

Hoàn thành200 nhưng GateB FAIL phải báo đúng số liệu, không đổi seed/reward/cap để tìm passing run. Metrics helper chưa implement thì các gate phụ thuộc nó UNVERIFIED, không thay bằng “đủ gần”. Exit0 không thay integrity; early interruption không phải completed C1.

## 15. Approval checklist

[x] chỉ xác nhận evidence tĩnh hiện tại, không phê duyệt execution. Recheck mọi item trước start.

- [x] Correct Git commit verified — fc46f39…; branch g1-multiskill.
- [x] Pinned software environment verified — source/metadata và Stage5/6 evidence; sáu known conflicts giữ nguyên.
- [ ] Exact current source config archived — hashes/current source và Stage5 snapshots có; pre-run external archive và genuine C1 resolved config chưa có.
- [x] Historical differences documented — C0 reproducibility BLOCKED.
- [ ] Training seed confirmed — đề xuất42, execution approval chưa nhận.
- [x] PPO hyperparameters confirmed — source và snapshot run mới, mục3.
- [x] Reward/curriculum/PD unchanged — xác nhận Stage8 không sửa; C1 runtime phải audit.
- [ ] Evaluation metrics and seeds approved —42/43/44, gates A/B/C chờ formal approval.
- [ ] Run output directory confirmed unique — future timestamp/genuine path chưa chốt.
- [ ] Disk/resource limits acceptable — hiện free168,72GiB;256env fit/time cap chưa verified/approved.
- [ ] External backup destination approved — **NOT SATISFIED**.
- [ ] Checkpoint backup strategy approved — mục12 vẫn là proposal.
- [ ] Estimated cost reviewed — rate/time UNKNOWN, chưa approved budget.
- [ ] Technical Lead approved — đã cho phép Stage8 design; execution approval chưa có.
- [ ] User explicitly authorized training — **CHƯA CÓ**.
- [ ] Evaluation instrumentation/offline metrics proposal reviewed — cần full metric coverage.
- [ ] Contact/support primary-gate use approved — chỉ tick sau C-T PASS; nếu chưa đạt, giữ secondary.

## 16. Final go/no-go recommendation

**STAGE 8 PASS WITH LIMITATIONS.**
**NOT READY — SPECIFIC BLOCKERS REMAIN.**

Đủ kế hoạch để review C1; hyperparameters/source/CLI đã audit, không training/simulation/config changes. Trước start cần external backup destination và restore/checksum procedure được duyệt, budget/rate-or-cap, approved metrics/evaluation instrumentation, unique paths/resources/config preflight. Missing C0 snapshots chặn exact historical evaluation, không nhất thiết chặn fresh C1; contact quality chưa đạt chỉ chặn dùng support margin làm primary gate.

Khi các blockers được xử lý, mới request Technical Lead và user training approval với scope C1/cost cap rõ ràng. Không ghi TRAINING APPROVED. Không tự chuyển execution, C2/C3/C4 hoặc Stage9. Dừng ở tài liệu này.
