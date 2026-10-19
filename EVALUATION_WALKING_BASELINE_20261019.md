# Đánh giá Walking Baseline G1 29 DoF — 2026-10-19

> Cập nhật sau chẩn đoán điều khiển cùng ngày: đã xác nhận cache angular velocity
> và projected gravity không được làm mới đầy đủ sau reset, ảnh hưởng returned
> observations của cả actor/critic. Các số liệu dưới đây vẫn mô tả pipeline lúc
> đo, nhưng không phải benchmark của một reset lifecycle đã đúng. Không quy toàn
> bộ việc ngã cho thiếu training hoặc reward trước khi sửa và kiểm chứng reset.
> Xem [báo cáo chẩn đoán và đối chứng sạch](CONTROL_DIAGNOSIS_AND_PLAN_20261019.md).

## Kết luận

Checkpoint `model_199.pt` load đúng và chạy được trên PhysX CUDA, nhưng **chưa đạt
Walking Baseline**. Robot chưa đứng vững, đi thẳng hoặc quay theo lệnh một cách
ổn định. Với lệnh đứng yên, episode trung bình chỉ **0,2597 giây**; tất cả episode
đã kết thúc đều do `bad_orientation`. Không robot nào giữ được episode đầu đến
2 giây trong bất kỳ tình huống của hai checkpoint được kiểm tra.

Checkpoint 199 cải thiện thời gian sống và sai số vận tốc tuyến tính so với
checkpoint 100, nhưng sai số quay không cải thiện đồng đều. Đối chứng action bằng
0 sống trung bình **1,2197 giây**, cũng không đứng ổn định lâu dài. Không nên xem
reward tăng hoặc loss giảm là bằng chứng robot đã biết đi.

Không chạy training mới, không sửa reward/curriculum/actuators và không thay
framework/dependency. Hash của hai checkpoint được kiểm tra trước/sau và không đổi.

## Artifacts để xem

- [Video robot trong Newton Viewer, 20 giây](run/evaluation-20261019/model_199_visible.mp4).
- [Ảnh robot trong viewer](run/evaluation-20261019/newton-follow-check.png).
- [Biểu đồ so sánh](run/evaluation-20261019/quantitative-final/evaluation-comparison.png)
  và [bản PDF](run/evaluation-20261019/quantitative-final/evaluation-comparison.pdf).
- [Diễn biến episode đầu](run/evaluation-20261019/quantitative-final/first-episode-traces.png).
- [CSV đầy đủ](run/evaluation-20261019/quantitative-final/summary.csv),
  [JSON đầy đủ](run/evaluation-20261019/quantitative-final/summary.json),
  [manifest](run/evaluation-20261019/quantitative-final/manifest.json).

Video là quay riêng cửa sổ Newton, 1280×672, 25 FPS, 500 frames. Viewer chạy khoảng
17 FPS nên thời gian video là thời gian màn hình; dùng số giây simulator trong
CSV để đánh giá thời gian sống. Robot bị reset khi nghiêng quá ngưỡng, trước khi
thân nằm hẳn trên mặt đất. Video không chứng minh robot đang đi ổn định.

## Checkpoint và cấu hình thực tế

Thư mục training:
`logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/`.

- 256 environments, 200 iterations, 1.228.800 transitions.
- Python 3.12; Isaac Sim 6.0.1.0; Isaac Lab source `v3.0.0-beta2.patch1`
  (distribution metadata 6.1.14); Torch 2.10.0+cu128; RSL-RL 5.0.1.
- G1 có 29 joint DoF và 29 actions. Viewer báo 30 joints vì có thêm floating root;
  đó không phải 30 khớp điều khiển.
- Actor 480 → 512 → 256 → 128 → 29; critic 495 → 512 → 256 → 128 → 1.
  History length 5, action scale 0,25 rad quanh vị trí khớp mặc định.
- Physics dt 0,005 s, decimation 4, control dt 0,02 s; episode limit 20 s.
- PPO: 24 steps/environment/rollout, 5 epochs, 4 minibatches, gamma 0,99,
  lambda 0,95, entropy coefficient 0,01, desired KL 0,01, adaptive learning rate.
- Initial command ranges: vx, vy, wz đều ±0,1. Command curriculum metric vẫn là
  0,1 ở cuối training. Các `limit_ranges` rộng hơn là giới hạn tương lai của
  curriculum, không có nghĩa policy đã được train đến vx = 1 m/s.
- Observation noise được bật. Ma sát/mass được randomize lúc startup; yaw lúc
  reset ở ±3,14 rad; joint velocity reset dùng hệ số [−1,1] nhân default velocity
  bằng 0 nên qdot khởi tạo thực tế bằng 0, không phải ±1 rad/s; push interval 5 s.
- Terrain generator có nhiều tile nhưng subterrain đang dùng là flat mesh.

Actor/critic của cả hai checkpoint đều hữu hạn. Evaluator xác minh tensor actor
sau `runner.load(..., strict=True)` khớp đúng checkpoint; chỉ inference, không gọi
`learn()`/optimizer update. Tham số độ lệch chuẩn Gaussian trung bình là 0,852 ở
checkpoint 100 và 0,653 ở checkpoint 199; inference dùng mean action, không sample
Gaussian. Chi tiết/hashes nằm trong
`run/evaluation-20261019/checkpoint-inspection.json`.

## Quy trình đo

Mỗi checkpoint: **64 robots × 20 giây simulator × 6 tình huống**. Thêm một đối
chứng action bằng 0 khi đứng yên. Tổng cộng **832.000 transitions**, **79.863
completed episodes**, 13 conditions, exit code 0.

Các trường cấu hình huấn luyện được khôi phục từ `params/env.yaml` và
`params/agent.yaml`, gồm tuple/slice trong SceneEntityCfg. Chỉ thay num_envs,
log_dir và đầu vào command của các tình huống cố định; tắt debug marker ở phần đo.
Reward, curriculum, observation noise, startup/reset/interval randomization và
termination vẫn giữ như training. Curriculum runtime vẫn hoạt động, không bị tắt.

Policy inference là deterministic theo API RSL-RL 5.0.1. Observation vẫn có noise,
nên đây không phải phép thử noiseless. Seed startup 42; reset seed là 42 cộng
chỉ số tình huống. Đối chứng zero-actions dùng reset seed 42, trong khi standing
của hai checkpoint dùng 43: so sánh này cùng cấu hình randomization nhưng không
phải paired comparison của từng trạng thái ban đầu.

Các command cố định được giữ cả sau auto-reset và resampling, có assert kiểm tra
toàn bộ command đã ghi. Tọa độ vận tốc tuyến tính theo thân robot, không phải
trục X cố định của thế giới; yaw reset vẫn ngẫu nhiên. Bài 0,5 m/s được đánh dấu
ngoài dải huấn luyện (OOD), tách khỏi các bài trong dải ±0,1.

Đo sau physics, **trước auto-reset** bằng callback của termination manager:

- Completed episode length lấy trực tiếp từ `episode_length_buf × 0,02` tại
  terminal step. Episode còn dang dở cuối cửa sổ không đưa vào trung bình.
- XY RMSE = sqrt(mean((vx−cmd_vx)² + (vy−cmd_vy)²)); yaw RMSE tương tự với wz.
- Velocity metrics dùng `root_lin_vel_b`/`root_ang_vel_b`, nhất quán với command
  metrics upstream. Reward tuyến tính dùng yaw frame, nên RMSE ở đây không phải
  phép đảo trực tiếp giá trị reward. Khi ngã, vx theo thân còn chịu ảnh hưởng của
  góc nghiêng; vx âm không đủ để kết luận robot đang đi lùi có kiểm soát.
- Bad-orientation fraction là số completed episodes có term đó / tổng completed
  episodes, không phải phần trăm timestep. Ngưỡng là **0,8 rad ≈ 45,84°**.
- Raw trajectories, commands và terminal flags được lưu NPZ cho từng tình huống.
  Các episode lặp lại trên 64 robots không được coi là hàng chục nghìn mẫu độc lập.

## Kết quả model_199

| Tình huống | Command (vx, vy, wz) | Episode TB (s) | XY RMSE (m/s) | Yaw RMSE (rad/s) | Bad orientation |
| --- | --- | ---: | ---: | ---: | ---: |
| Commands như training | Ngẫu nhiên trong ±0,1 | 0,2591 | 0,2376 | 0,4344 | 100% |
| Đứng yên | (0, 0, 0) | 0,2597 | 0,2265 | 0,4305 | 100% |
| Đi thẳng chậm | (0,1; 0; 0) | 0,2599 | 0,3141 | 0,4130 | 100% |
| Quay trái | (0; 0; 0,1) | 0,2602 | 0,2221 | 0,4894 | 100% |
| Quay phải | (0; 0; −0,1) | 0,2585 | 0,2307 | 0,3845 | 100% |
| Đi thẳng OOD | (0,5; 0; 0) | 0,2627 | 0,6932 | 0,3561 | 100% |

Các episode dài nhất của model_199 chỉ 0,28 s (riêng quay phải tối đa 0,26 s).
Không có episode chạm timeout 20 s hoặc kết thúc vì root height < 0,2 m: ngưỡng
nghiêng được chạm trước. Với standing: 4.898 completed episodes, tilt trung bình
18,29°, p95 49,07°, angular XY RMS 3,64 rad/s, phần timestep tilt < 20° là 61,62%.
Con số 61,62% chỉ phản ánh nhiều đoạn ngay sau reset, không phải standing success.

## So sánh checkpoint 100 và 199

| Tình huống | Episode 100 (s) | Episode 199 (s) | XY RMSE 100 → 199 | Yaw RMSE 100 → 199 |
| --- | ---: | ---: | --- | --- |
| Training commands | 0,1584 | 0,2591 | 0,5486 → 0,2376 | 0,3921 → 0,4344 |
| Đứng yên | 0,1590 | 0,2597 | 0,5415 → 0,2265 | 0,3894 → 0,4305 |
| Đi chậm | 0,1549 | 0,2599 | 0,6530 → 0,3141 | 0,3922 → 0,4130 |
| Quay trái | 0,1597 | 0,2602 | 0,5361 → 0,2221 | 0,3104 → 0,4894 |
| Quay phải | 0,1576 | 0,2585 | 0,5477 → 0,2307 | 0,4801 → 0,3845 |
| Đi OOD 0,5 | 0,1401 | 0,2627 | 1,0983 → 0,6932 | 0,3437 → 0,3561 |

Standing episode của 199 dài hơn khoảng 63%, XY RMSE giảm khoảng 58%, nhưng cả
hai checkpoint đều ngã rất sớm. Không gọi model_199 là policy đi bộ tốt hơn toàn
diện vì một số yaw errors tăng.

Đối chứng zero-actions: completed episode TB 1,2197 s, p95/max 1,24 s, angular XY
RMS 0,971 rad/s, yaw RMSE 0,0093 rad/s. Tuy nhiên XY RMSE 0,5675 m/s và 100%
episodes vẫn bad_orientation. Zero action không phải controller đứng cân bằng;
so sánh này chỉ cho thấy policy học được hiện chưa duy trì được tư thế tốt hơn
một nominal-pose reference về thời gian sống.

## Tại sao còn ngã: bằng chứng và điều chưa kết luận được

1. **Policy chưa học được feedback giữ thăng bằng.** Tilt tăng đến terminal
   threshold trong 0,26 s. Ở standing, deterministic action RMS là 0,974,
   tương đương RMS target offset khoảng 0,244 rad (~14°) với scale 0,25.
   Joint pose deviation RMS đo được 0,178 rad. Đây là chuyển động do mean policy,
   không phải do sampling action noise trong evaluator.
2. **Learning còn rất sớm.** Từ model_100 đến 199 có tiến bộ về thời gian sống,
   nhưng training chỉ chạy khoảng 3,5 phút/200 iterations. Không có cơ sở kỳ vọng
   200 iterations tự bảo đảm walking. Nhiều dữ liệu training là các đoạn reset
   rất ngắn; reward tăng từ −1,197 lên −0,453 trong TensorBoard không thay đổi
   kết luận stability. Mean episode length training cuối 13,43 steps ≈ 0,269 s,
   phù hợp với evaluation, không phải 13,43 giây.
3. **Bài 0,5 m/s chưa thuộc distribution đã học.** Curriculum vẫn ở 0,1. Tuy vậy,
   policy cũng thất bại ngay với zero command và 0,1, nên OOD không giải thích
   toàn bộ vấn đề.
4. **Reward balance cần kiểm chứng bằng A/B.** Alive weight 0,15; action-rate
   penalty −0,05; orientation penalty −5; feet-clearance reward +1. Foot-clearance
   không bị gate theo command trong config hiện tại. Logs cuối có alive +0,00184,
   action-rate −0,01434, orientation −0,01020, clearance +0,01251 theo chuẩn hóa
   logging của reward manager. Chúng gợi ý cân bằng survival/motion cần xem lại,
   nhưng chưa chứng minh một reward term đơn lẻ gây ra ngã hay policy cố tình ngã.
5. **Không nên đổ lỗi ngay cho pushes hoặc framework.** Các policy ngã trước
   push 5 giây; mean inference đã ngã khi không sample Gaussian. Đồng thời zero
   reference cũng không đứng bền, nên tư thế/PD/randomization cần ablation riêng.
   Chưa thực hiện các ablation đó, chưa có bằng chứng phải đổi backend, framework
   hay mapping khớp.

Built-in command success thresholds là 0,5 m/s và 0,4 rad/s, rộng hơn dải command
ban đầu ±0,1, và không yêu cầu sống đủ lâu. Vì vậy success_rate dương trên một
episode rất ngắn không có nghĩa robot đã biết đi. Evaluator còn xuất tracking pass
với ngưỡng chặt hơn 0,1 m/s / 0,2 rad/s, nhưng vẫn phải xét survival riêng.

## Đề xuất cho bước tiếp theo — chưa áp dụng

Ưu tiên kiểm chứng từng giả thuyết với một biến thay đổi mỗi lần:

1. Dùng bài đứng/đi rất chậm làm gate: sống ít nhất 10–20 s, tilt và tracking error
   đạt ngưỡng đã thống nhất trước khi tăng command ranges. Giữ ngưỡng ngã 0,8 rad;
   không nới termination để làm episode length đẹp hơn.
2. Ablation đánh giá trước: giữ nguyên policy, lần lượt tắt observation noise,
   mass/friction randomization, rồi kiểm tra settling/nominal pose/PD
   trên nền phẳng. Chúng chỉ là thí nghiệm đề xuất, chưa chạy hoặc sửa config.
3. Với một nhánh training được phê duyệt: ưu tiên upright/survival, cân nhắc
   terminal fall penalty theo reward scale; giảm/gate gait và clearance khi học
   đứng rồi tăng dần lúc chuyển sang walking. So sánh bằng survival và tracking,
   không chỉ mean reward.
4. PPO: thử init action std khoảng 0,3–0,5 thay vì 1,0 cho nhánh đứng ban đầu;
   cân nhắc rollout 32–48 steps để quan sát đoạn dài hơn. Đây là giá trị khởi đầu
   cho A/B, không phải mức tối ưu đã chứng minh. Gaussian std cuối hiện đã giảm
   còn 0,653; deterministic inference vẫn thất bại nên giảm std đơn thuần chưa đủ.
5. Không tự tăng/giảm learning rate một cách mù quáng: dù config đặt 1e−3,
   adaptive scheduler thực tế đã đưa LR xuống 1e−5 lúc đầu và khoảng 1,14e−4 ở cuối.
   Cần xem KL và các metric stability trước khi chỉnh PPO thêm.
6. Nếu được phép train tiếp, dùng đợt giới hạn và checkpoint định kỳ, đánh giá
   cùng protocol trước khi quyết định kéo dài. Không chạy thêm bất kỳ iteration
   training nào trong nhiệm vụ đánh giá này.

Các đề xuất là suy luận từ phép đo local và cấu hình hiện có. Tham khảo nền tảng
thuật toán: [PPO paper](https://arxiv.org/abs/1707.06347); mã baseline gốc nằm trong
`.deps/unitree_rl_lab-multiskill` ở commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`.

## Lệnh playback đã kiểm chứng

```bash
cd /workspace/Humaniod-MultiSkill
export OMNI_KIT_ACCEPT_EULA=YES
./run_multiskill.sh scripts/rsl_rl/play.py \
  --task G1-MultiSkill-v0 --num_envs 1 \
  --checkpoint logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_199.pt \
  --visualizer newton --kit_first --skip_export --camera_env 0 \
  --max_steps 1000 --real-time
```

Đã chạy 500 steps headless và 1.000 steps với camera Newton đúng vị trí, exit 0.
`--camera_env` cần thiết vì terrain tile của env_0 nằm xa world origin. `--skip_export`
bỏ công việc export ONNX/JIT không cần cho inference. Có thể đổi sang `model_100.pt`
để xem checkpoint trước. Headless: dùng `--visualizer none`, bỏ `--camera_env`.

Lệnh đo đầy đủ:

```bash
./run_multiskill.sh scripts/rsl_rl/evaluate_multiskill.py \
  --checkpoints \
    logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_100.pt \
    logs/rsl_rl/g1_multiskill_baseline/2026-10-19_08-46-28/model_199.pt \
  --num_envs 64 --seconds 20 --visualizer none --zero_reference \
  --output run/my-walking-evaluation
.venv-multiskill/bin/python -B scripts/rsl_rl/summarize_multiskill_eval.py \
  --input run/my-walking-evaluation
```

## File sửa/thêm trong nhiệm vụ này

- `scripts/rsl_rl/play.py`: thêm early Kit startup, skip-export, camera follow.
- `scripts/rsl_rl/evaluate_multiskill.py`: evaluator chỉ inference, terminal-state
  measurements và raw NPZ.
- `scripts/rsl_rl/evaluation_io.py`: đọc snapshot YAML, cho phép rõ ràng tuple/slice.
- `scripts/rsl_rl/summarize_multiskill_eval.py`: CSV/JSON và PNG/PDF.
- `README_MULTISKILL_VM.md`, báo cáo này và artifacts dưới `run/evaluation-20261019/`.

Các lỗi phát triển harness (YAML slice loader, strict updater với optional None,
reset history buffer ngoài InferenceMode) đã được xử lý trong helper evaluation,
không sửa framework. Lần chạy cuối `quantitative-final` hoàn tất cả 13 conditions.
Các log thử chưa hoàn tất và video camera world-origin trống được giữ làm
diagnostic; chỉ các artifacts được liên kết ở đầu báo cáo là deliverables cuối.

Còn cảnh báo gRPC descriptor/deprecation và debug marker rendering; không chặn
các bài inference đã kiểm chứng. Đây là một seed startup với randomization, trên
flat tiles, chưa phải đánh giá nhiều seed hoặc generalization toàn diện. Không
thay đổi checkpoint, reward, curriculum, trainer hay các bản sửa startup đã giúp
training thành công; không commit/push.
