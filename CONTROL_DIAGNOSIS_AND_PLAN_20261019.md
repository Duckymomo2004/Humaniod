# Chẩn đoán điều khiển và kế hoạch Walking Baseline — 19/10/2026

## Kết luận có bằng chứng

**Có lỗi cache trạng thái sau reset trong Isaac Lab PhysX hiện hành.** PhysX đã
đặt robot về tư thế đứng và vận tốc gốc bằng 0, nhưng observation đầu episode mới
còn chứa angular velocity và projected gravity của trạng thái ngã trước đó. Lỗi
xảy ra ở actor và critic, trong đường auto-reset dùng chung với training. Cần sửa
và kiểm chứng vấn đề này trước khi điều chỉnh reward hoặc chạy thêm PPO.

**Lỗi reset không giải thích toàn bộ việc ngã.** Trong episode đầu khởi tạo sạch,
cùng pose, velocity, joints, mass, friction và observations, model_199 vẫn ngã
trung bình 0,250 s, zero-actions là 1,2175 s. Mean action của checkpoint hiện tại
gây chuyển động mất cân bằng ngay trước khi có bất kỳ auto-reset nào.

Chưa phát hiện lỗi thứ tự joint, action offset/scale, normalization hoặc stochastic
inference. Đường truyền action đúng không có nghĩa posture/PD gains đã tối ưu.
Zero-actions cũng không đứng bền.

**Không sửa trainer, training config, framework, checkpoint, reward, curriculum,
PD gains hoặc dependency trong lượt này. Không chạy training mới.** Cache refresh
và action gain chỉ được áp dụng trong tiến trình chẩn đoán, chưa tích hợp vào trainer.

## Phạm vi và bảo toàn

- Branch `g1-multiskill`, RTX 3090, Python 3.12, Sim 6.0.1.0, torch 2.10.0+cu128,
  RSL-RL 5.0.1. Isaac Lab giữ tag `v3.0.0-beta2.patch1`, commit
  `ffff603eafc6b74264a5261cc0183d6a65390d78`; pip metadata ghi `isaaclab=6.1.14`.
- Checkpoint: `logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_199.pt`,
  SHA256 `418fd00f1c20c8d6ada52746943471cb66e50f2ad5afc354d8bff5ee35a578d6`.
- Khôi phục `params/env.yaml` và `params/agent.yaml` của run training. Dùng 8 robots,
  command `(0,0,0)`, physics dt 0,005 s, decimation 4, control dt 0,02 s.
  Giữ observation noise, mass/material randomization, reset/interval events,
  rewards, termination và curriculum. Chỉ thay num_envs, command và logging cho diagnostic.
- 51 file quan trọng gồm model_100/199, trainer, evaluator và config được đối chiếu
  SHA256 với snapshot trước chẩn đoán; xem `run/control-diagnosis-20261019/preservation_check.json`.
  Isaac Lab checkout không đổi; không commit/push, không xóa kết quả cũ.

## A1. Articulation, action và PD

**29 DoF, 29 controlled joints, action dimension 29.** Actor dùng đúng thứ tự
simulator/action term và danh sách joint lưu ở training. Thứ tự SDK khác simulator,
nhưng đường PPO hiện tại không reorder theo SDK nên không có bằng chứng tráo khớp.

[CSV đủ 29 outputs, SDK index, default pose, limits và PD](run/control-diagnosis-20261019/fresh_seed_42/gain_1p0/joint_mapping.csv).
Đã probe từng output bằng one-hot mà không bước physics: sai số affine bằng 0.
Target thực nhận ở PhysX mỗi control step cũng khớp chính xác.

`q_target = q_default + 0.25 × action`, q tính bằng radian, không clip action.
Gain α trong diagnostic là `q_default + 0.25 × α × policy_mean`, không sửa scale config.

| Nhóm joint | Kp | Kd | Torque limit (Nm) | Default q (rad) |
| --- | ---: | ---: | ---: | --- |
| Hip pitch/yaw | 100 | 2 | 88 | pitch −0,1; yaw 0 |
| Hip roll | 100 | 2 | 139 | 0 |
| Knee | 150 | 4 | 139 | 0,3 |
| Waist yaw | 200 | 5 | 88 | 0 |
| Waist roll/pitch | 40 | 5 | 25 | 0 |
| Ankle pitch/roll | 40 | 2 | 25 | pitch −0,2; roll 0 |
| Shoulder, elbow, wrist roll | 40 | 1 | 25 | xem CSV |
| Wrist pitch/yaw | 40 | 1 | 5 | 0 |

Tất cả là `ImplicitActuator`. Stiffness, damping, maximum force đọc trực tiếp từ
PhysX khớp asset data. Torque log gồm PD estimate và incoming joint force chiếu
lên trục joint. **Không gọi incoming projection là motor torque thuần**; xem
[NVIDIA articulation force sensors](https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_physics_articulation_force.html).
Trong đoạn trước env0 terminal của trace 8 robots, PD estimate không vượt effort
limit; chưa có bằng chứng torque saturation là nguyên nhân chính.

## A2. Observation và deterministic inference

- Actor 480 = 96 × 5 history: angular velocity, gravity, command, relative q,
  relative qdot, last action. Critic 495 = 99 × 5, thêm linear velocity.
- History nối theo term, không phải 5 full frames. Đã so từng block với circular
  buffer thật, khớp chính xác. Checkpoint load strict actor/critic; actor
  480→512→256→128→29, normalizer `Identity`, không thiếu running statistics.
- Actor eval; forward mặc định và `stochastic_output=False` bằng nhau, lặp lại
  bằng nhau và không đổi CUDA RNG. Không sample Gaussian exploration.
  Observation noise vẫn bật giống training; deterministic policy không có nghĩa
  observations hết noise.
- Cấu trúc history đúng nhưng nội dung sau reset bị cache làm sai. Chưa thể coi
  state/observation lifecycle của training và inference là hoàn toàn đúng.

## A3. Lỗi reset và phép thử A/B

Auto-reset đầu trong `returned_obs_baseline`, lúc 0,24 s:

| Đại lượng trên robots vừa reset | Hiện tại | Refresh cache trong diagnostic |
| --- | ---: | ---: |
| Max absolute native root velocity | 0 | 0 |
| Sai số angular history critic, bỏ scale 0,2 (rad/s) | 5,50979 | 0 |
| Sai số gravity history critic | 0,74555 | 2,38e−7 |
| Sai số angular history actor sau scale | 1,09398 | 0,03419 |
| Sai số gravity history actor | 0,75152 | 0,04940 |

Actor noise bounds: angular sau scale ±0,04, gravity ±0,05. Sau refresh, sai số
nằm trong các giới hạn này. Critic không noise nên so trực tiếp được. Đây là
observations **đã trả về từ `wrapped.step()`**, không chỉ cache đọc giữa reset.

Định vị ở commit hiện hành:

- `isaaclab_physx/.../assets/articulation/articulation.py`,
  `write_root_link_pose_to_sim_index` khoảng dòng 445–505 ghi pose nhưng chưa
  invalidate projected gravity, heading và các derived state phụ thuộc orientation.
- Cùng file, `write_root_com_velocity_to_sim_index` khoảng dòng 680–733 ghi đúng
  velocity vào PhysX/root COM buffer nhưng chưa invalidate root-link velocity
  và các cache velocity suy ra.
- `articulation_data.py`: getter derived chỉ cập nhật khi timestamp cũ hơn simulation.
  Reset cùng timestamp giữ dữ liệu derived từ trạng thái trước reset.
- `manager_based_rl_env.py`: termination/reward đọc gravity/angular velocity
  trước reset, rồi auto-reset và observation trong cùng step. Observation đọc lại
  cache đó; history đã reset lấp observation đầu vào cả 5 slots, mang giá trị sai.

`--refresh_reset_cache` chỉ invalidate timestamp 8 derived buffers sau `_reset_idx`,
không ghi physics state. Native velocity trước/sau không đổi. Multi-reset seeds
42/43/44 khớp lại paired initial states và observations; angular error sau refresh
bằng 0, gravity cỡ 1e−7. **Đây là proof of cause, chưa là bản vá framework đầy đủ**:
cần regression full/subset reset, pose-only, velocity-only, index/mask và non-reset envs.

Run `paired` đầu tiên chủ động FAIL assertion: poses/joints/randomization khớp,
nhưng cached velocity/observations của các reset sau phụ thuộc rollout trước.
Không dùng run đó để kết luận action gain; giữ log FAIL. `fresh_seed_42` chỉ dùng
reset đầu mỗi env mới, pairing pass sai số 0, không cache refresh.

## A4. So sánh cùng điều kiện và 0,5 giây đầu

8 robots, startup seed 42, reset seed 42; tất cả initial states và observations
giữa các nhánh khớp tuyệt đối. Đo first failure trước auto-reset, cùng ngưỡng
`bad_orientation=0,8 rad`, root height 0,2 m. Dữ liệu: `fresh_seed_42/manifest.json`
và `results.json`.

| Action đưa vào simulator | First failure TB (s) |
| --- | ---: |
| Model_199, α=1 | 0,2500 |
| α=0,5 | 0,8625 |
| α=0,25 | 1,0550 |
| Zero-actions, α=0 | 1,2175 |

Tất cả fail trước 1,5 s. Giảm amplitude làm chậm mất cân bằng trong đối chứng này;
không chứng minh α=0,25 là controller đứng hay scale tối ưu. Đây là một startup
seed với 8 robots, không phải 8 independent training seeds. Run cache refresh
qua 3 reset seeds cho model_199 lần lượt 0,2500; 0,2475; 0,2525 s; zero đều 1,2175 s.
Làm sạch cache đơn thuần không cứu được checkpoint hiện tại.

Trace `first_half_second` ghi 26 mốc 0…0,5 s: root pose/velocity, roll/pitch,
COM toàn robot theo mass, foot positions/forces, 29 q/qd/targets/actions/torque.
**Chỉ trace này chặn auto-reset để nhìn liên tục 0,5 s**; terminal gốc vẫn được
tính. Đoạn sau terminal tô nền, không tính là sống thêm. Các benchmark ở bảng trên
giữ auto-reset bình thường.

Ví dụ env0:

| Thời điểm | Pitch model | Pitch zero | Foot Fz model L/R (N) | Foot Fz zero L/R (N) |
| --- | ---: | ---: | --- | --- |
| 0,02 s | 0,82° | ~0° | 0 / 0 | 0 / 0 |
| 0,06 s | 3,33° | −0,11° | 411 / 0 | 1529 / 1479 |
| 0,10 s | 7,09° | −1,68° | 386 / 0 | 199 / 200 |
| 0,20 s | 30,31° | −2,22° | 222 / 45 | 172 / 181 |
| 0,26 s | 48,80° — terminal | −1,63° | 236 / 72 | 173 / 172 |
| 0,50 s | 59,37° — sau terminal | 1,88° | 81 / 264 | 182 / 173 |

Action đầu làm knee targets L/R thành 0,1492/0,3853 rad thay vì 0,3/0,3; ankle
pitch thành −0,5126/−0,3609 rad thay vì −0,2/−0,2. Hai chân nhận lệnh khác nhau,
touchdown khác nhau, pitch tăng nhanh. Đây là bất đối xứng mean policy, không phải
mapping hoán đổi; cần ablation từng nhóm để quy kết riêng joint nào.

COM env0 ban đầu z=0,72553 m. Đến 0,26 s, model COM lệch X/Y theo initial yaw
+0,0062/−0,0290 m, giảm z 0,0242 m; zero +0,0149/+0,0002 m, giảm z 0,0222 m.
COM khác root position. Hai điểm ankle chưa đủ xác định support polygon/contact
patch, không suy ra support margin chính xác từ hai điểm này.

Ban đầu hai chân chưa báo contact; zero rơi ngắn, touchdown lúc 0,06 s với COM
giảm 1,66 cm. Nominal pose chưa phải trạng thái cân bằng đã settle. Cần thử riêng
spawn height/settling trước khi kết luận PD sai. Joint velocity khởi tạo đo được
**bằng 0**: `reset_joints_by_scale` nhân `default_joint_vel=0` với [−1,1], không
phải random qdot ±1 rad/s.

- [Roll/pitch, COM, velocity, contacts, actions](run/control-diagnosis-20261019/analysis/first_half_second.png)
- [Joint q/target, torque và 29 outputs](run/control-diagnosis-20261019/analysis/joint_control.png)
- Bản PDF, CSV toàn bộ 29 joints × 8 robots × 26 mốc mỗi nhánh và summary JSON
  nằm trong `run/control-diagnosis-20261019/analysis/`.

## Phân loại nguyên nhân

| Loại | Kết luận |
| --- | --- |
| Mapping/action transport | Các probe pass; chưa có bằng chứng cần reorder hay đổi scale |
| Reset/observation | Lỗi được xác nhận bằng native state, returned obs và A/B |
| Đánh giá | Mean inference đúng; run không paired bị loại, có đối chứng sạch thay thế |
| Policy hiện tại | Mean action gây mất cân bằng ngay first episode sạch |
| Chưa học đủ | Có thể, nhưng không quy hết cho thiếu iterations khi reset đang sai |
| Posture/PD/settling | Zero cũng ngã; cần ablation riêng, chưa chứng minh gain sai |
| Reward/curriculum | Cần A/B sau reset fix; chưa chứng minh term nào gây ngã |

## B. Curriculum và reward đề xuất — CHƯA ÁP DỤNG

Điều kiện trước tiên là reset regression pass. Có thể tham khảo cách tăng dần độ
khó của [Rudin et al.](https://arxiv.org/abs/2109.11978); các ngưỡng dưới đây là
thiết kế thí nghiệm cho G1 này, chưa là kết quả tối ưu.

| Giai đoạn | Phạm vi đề xuất | Gate chuyển giai đoạn đề xuất |
| --- | --- | --- |
| Standing/balance | Plane, command zero; perturbation sau khi kiểm chứng settle | ≥90% sống đủ 20 s, tilt p95 <20°, ≥3 eval seeds |
| Đi chậm | vx 0,05–0,15 m/s, vy/yaw=0, trộn standing | ≥90% sống 20 s, XY RMSE <0,10 m/s |
| Tốc độ và quay | Mở vx rồi vy rồi yaw từng trục; ví dụ vx tới 0,5, yaw ±0,2 | ≥90% sống 20 s, XY RMSE <0,15 m/s, yaw RMSE <0,15 rad/s |
| Địa hình khó | Nhám nhỏ, dốc, rồi bậc; mỗi lần một loại | Survival giảm không quá 5 điểm % so với plane, kiểm terrain holdout |

Terrain hiện tên cobblestone nhưng thực tế chỉ có sub-terrain flat plane.
Curriculum levels không tạo rough terrain khi generator chưa có loại rough.

Các reward changes phải thử từng yếu tố, log raw term và weighted term:

1. Standing: giữ orientation (hiện −5), base-height (−10), zero-velocity tracking,
   action-rate (−0,05), deviation arms/waist. Đo root height sau settle trước khi
   quyết định sửa target 0,78 m; không so trực tiếp target root với COM. Thử tăng
   survival emphasis (alive hiện +0,15) **hoặc** thêm penalty cho fall termination,
   mỗi thay đổi một A/B; chọn mức theo discounted return/log scale, timeout không phạt.
2. Thử riêng gate feet-clearance theo command/swing phase hoặc weight=0 khi
   standing. **Gait đã gate command norm >0,1**. Clearance chưa gate command,
   nhưng có `tanh(foot_xy_speed)` và reward gần 1 khi speed=0, nên chưa có cơ sở
   nói term này ép nhấc chân lúc đứng. Không kết luận nguyên nhân chỉ từ weight.
3. Walking: điều chỉnh gait/clearance sau standing pass. Gate hiện >0,1 khiến
   command 0,05–0,1 không có gait reward; thay gate là thí nghiệm riêng. Giữ slide
   và undesired-contact penalties, theo dõi contact timing.
4. Rough: kiểm reward độ cao tương đối mặt đất; clearance hiện dùng foot world-z
   với target cố định 0,1 m. Không chuyển nguyên công thức đó sang bậc cao chưa kiểm chứng.
5. Termination trước mắt giữ góc 0,8 rad, root height 0,2 m, timeout 20 s làm chuẩn.
   Nếu bổ sung body-contact termination, đo contact/settling và thử riêng. Không
   nới góc để tăng episode length; giữ timeout là time-limit cho PPO.

## Thí nghiệm tiếp theo — CHƯA CHẠY

1. **Reset regression, không training:** fix cache có thể hoàn tác trong nhánh
   local riêng. Test full/subset, pose-only/velocity-only, setters index/mask,
   actor/critic history, native state unchanged, non-reset envs liên tục. Nhiều
   episodes model_199; critic error <1e−5, actor error trong noise bounds.
2. **Control ablation, không training:** zero-action cùng initial states, chỉ
   thay settling/spawn một lần; nếu cần mới thử PD từng nhóm trong lượt riêng.
   Đo contacts/COM/height/survival. Không đổi reward/PPO cùng lượt.
3. **Reset-corrected PPO pilot:** giới hạn 256 env × 200 iterations, cùng seed,
   initialization và config baseline, chỉ khác reset fix. Lưu run mới; đánh giá
   standing và vx=0,1 trên ≥3 seeds, first/all-episode metrics riêng. Baseline cũ
   là tham chiếu; nếu cần A/B training độc lập, phải cân nhắc chi phí và được yêu
   cầu trước, không tự lặp lại training lỗi.
4. Nếu vẫn chưa đứng: nhánh standing mới, trước tiên chỉ zero commands; các
   reward changes vào từng thí nghiệm sau đó, so với standing reference cố định.
   Mỗi pilot tối đa 200 iterations, stop/report nếu survival không tăng; không
   tự kéo lên 3.000 iterations.
5. Sau đó mới thử init action std 0,3–0,5 thay 1,0, hoặc rollout dài hơn, mỗi lần
   một yếu tố. Mean inference hiện đã ngã nên giảm exploration std không sửa
   trực tiếp checkpoint 199. Chỉ tăng training seeds khi có ứng viên pass standing.

## File mới/sửa và tái hiện

Thêm `scripts/rsl_rl/diagnose_multiskill_control.py`,
`scripts/rsl_rl/summarize_control_diagnosis.py`, báo cáo này và phần đính chính
trong báo cáo evaluation cũ. Trace mới ở `run/control-diagnosis-20261019/`.
Các sửa startup/evaluation có sẵn từ trước không đổi trong lượt chẩn đoán này.

Từ `/workspace/Humaniod-MultiSkill`, dùng output mới để giữ kết quả hiện có:

```bash
OMNI_KIT_ACCEPT_EULA=YES ./run_multiskill.sh scripts/rsl_rl/diagnose_multiskill_control.py \
  --checkpoint logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_199.pt \
  --num_envs 8 --seeds 42 --gains 1 0 0.5 0.25 --seconds 1.5 \
  --visualizer none --output run/control-diagnosis-rerun-fresh

OMNI_KIT_ACCEPT_EULA=YES ./run_multiskill.sh scripts/rsl_rl/diagnose_multiskill_control.py \
  --checkpoint logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_199.pt \
  --num_envs 8 --seeds 42 43 44 --gains 1 0 --seconds 1.5 \
  --refresh_reset_cache --visualizer none --output run/control-diagnosis-rerun-cache

OMNI_KIT_ACCEPT_EULA=YES ./run_multiskill.sh scripts/rsl_rl/diagnose_multiskill_control.py \
  --checkpoint logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_199.pt \
  --num_envs 8 --seeds 42 --gains 1 0 --seconds 0.5 --no_autoreset \
  --visualizer none --output run/control-diagnosis-rerun-half-second

.venv-multiskill/bin/python scripts/rsl_rl/summarize_control_diagnosis.py \
  --root run/control-diagnosis-20261019
```

Không dùng `--no_autoreset` cho survival benchmark/training. `--refresh_reset_cache`
chưa tích hợp vào train/play/evaluator chính. Chưa có kết quả robot đứng hoặc đi
được; đây là chẩn đoán có đối chứng và kế hoạch sửa/test tiếp theo.
