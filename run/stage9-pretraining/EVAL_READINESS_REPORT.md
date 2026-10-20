# Stage 9 — C1 evaluation readiness và backup manifest

## A. Executive summary

**STAGE 9 OFFLINE PASS — RUNTIME VERIFICATION PENDING**

- **OFFLINE EVALUATOR PASS**, **SUMMARIZER PASS**, **BACKUP MANIFEST PASS**, **INTEGRITY PASS** trong phạm vi đã kiểm.
- **BACKUP GATE: NOT SATISFIED**
- **BUDGET GATE: NOT SATISFIED**
- **TRAINING APPROVAL: NOT GRANTED**
- Không Isaac Sim, PhysX, PPO, GPU benchmark hoặc inference thật. Không upload/commit/push.
- Branch g1-multiskill, HEAD fc46f39ef0214e84cb0bb484dfec0d1fef7ba161.

Tạo evaluator và summarizer riêng:
[evaluate_multiskill_c1.py](../../scripts/rsl_rl/evaluate_multiskill_c1.py),
[summarize_multiskill_c1.py](../../scripts/rsl_rl/summarize_multiskill_c1.py),
[unit tests](../../scripts/rsl_rl/test_multiskill_c1_offline.py).
Các evaluator/summarizer gốc, trainer, environment, reward/curriculum/PD và báo cáo Stage7/8 giữ nguyên.

Đọc [kế hoạch Mốc C](../../MOC_C_EXPERIMENT_PLAN.md), [báo cáo Mốc B](../../MOC_B_FINAL_REPORT.md), scripts gốc, reset test, posture geometry, nguồn G1, TerminationManager và RSL-RL5.0.1, hai YAML thật Stage5 smoke64. Source read inventory và hash nằm trong [static_audit.json](static_audit.json). Không dùng YAML Stage5 thay lịch sử model_199.

## B. Implementation và audit API

### B1. Dựa trên pipeline hiện có

Giữ cách restore cfg.from_dict từ YAML cùng run, AppLauncher bootstrap trước task/config imports, gym.make G1-MultiSkill-v0, RslRlVecEnvWrapper, command resample/update hooks và capture qua TerminationManager.compute **sau physics/trước auto-reset** của evaluator gốc.

Khác biệt có chủ ý: import simulator/Torch/NumPy chỉ trong run_runtime(); import module/--help/tests chỉ standard library. Mỗi condition tạo fresh scene và reset bằng đúng --seed, không cộng scenario_index. Output root/condition/chunk/summary được tạo exclusive, không overwrite.

Snapshot configuration overrides chỉ ở evaluator local: num_envs, seed, log_dir, tắt debug marker/visualizer, fixed command inputs theo scenario. Reward/curriculum/PD/posture/action scale/noise/events/termination giữ cấu hình run thật. Không có writer vào production config.

RSL-RL source đã kiểm:
- OnPolicyRunner.get_inference_policy chuyển eval mode.
- MLPModel.forward(..., stochastic_output=False) trả deterministic distribution output.
- PPO.load nhận actor/critic/optimizer/iteration flags; evaluator load actor/critic strict từ torch.load(weights_only=True), **optimizer=False**, không learn/update/backward/optimizer.step.
- Runner constructor vẫn có thể tạo algorithm/optimizer objects để dùng API tương thích; điều đó không phải optimizer update.

### B2. Timeout API đã đọc, không đoán tên

Pinned source TerminationManager cung cấp active_terms, get_term(), get_term_cfg().time_out, terminated, time_outs và dones. Adapter kiểm aggregate OR của individual flags khớp terminated/time_outs/dones, rồi mới phân loại. Không mặc định mọi done là ngã, không hardcode timeout chỉ từ chuỗi tên.

Các term hiện tại là bad_orientation, base_height và time_out; các term không có time_out=true khác được giữ như other physical reasons. Source time_out dùng episode_length_buf >= max_episode_length; max_episode_length=ceil(20/0,02)=1000. Hai cờ physical và timeout đồng thời được phân loại **physical failure**, vẫn giữ timeout flag để audit.

### B3. Metrics và schema

Schema **g1-c1-evaluation-v1** gồm:
- invocation.json: input hashes/CLI/resource budgets, không credentials.
- condition.json: COMPLETE chỉ sau đủ rollout/stream/chunks, actual seed/horizon/dt/versions, input/config hashes, initial fingerprints, local overrides.
- initial_state.npz: relative root pose, velocity, q/qdot, policy/critic obs, mass/inertia/material, Kp/Kd/limits, commands, COM/origins.
- chunk-*.npz: bounded post-physics telemetry, global control time, episode steps, root quaternion/pose/velocities, RPY/tilt/height, COM, full actions/q/qdot/targets, torque signals, foot Fz và termination flags.
- episodes.jsonl: online episode records, ghi ngay khi terminated/truncated; có horizon-censored tails nếu còn samples.
- summary.json: first episodes riêng, later/all episode counts riêng. Summarizer recompute từ episodes và verify checksums, không tin summary có sẵn.

| Nhóm metric | Hỗ trợ offline/adapter |
|---|---|
| First episode | Physical event time hoặc None, observed duration, reasons, time-limit flag, horizon censor |
| Survival | S2/S5/S10/S20; early censor tạo UNKNOWN với lower/upper bounds, không drop |
| Duration | Observed mean/median/p10 có nhãn; physical median/p10 bằng KM, NOT_REACHED nếu censored |
| Orientation | Signed roll/pitch trong chunks; absolute roll/pitch và tilt p95 upper-bin |
| Root | Quaternion, position, height, body linear/angular velocity; RMS/min/max |
| COM | Whole-body mass-weighted COM, max 3D displacement từ initial COM |
| Tracking | XY/yaw RMSE time/sample-weighted, exposure; per-env RMSE distribution và duration |
| Action | RMS và delta-action RMS/controlstep; reset boundary không nối delta |
| Joints | Target−q RMS/peak theo đủ29 joints; qdot RMS và raw29 values trong chunks |
| Torque | Unclipped computed PD estimate, clipped applied estimate, effort-limit exceedance samples; projected reaction giữ riêng |
| Foot loads | L/R sampled net Fz, first load sample time, asymmetry khi denominator hợp lệ |
| Contact hull/CoP | Pure quality classifier CPU có tests; runtime raw hull **MISSING_TELEMETRY**, không primary gate |

PD computed/applied là implicit-actuator estimates, có timing khác native solver torque; incoming projection là reaction, **không gọi là motor torque measurement**. API projected-force getter được dùng trong diagnostic cũ và có pinned mock signature; native binding/shape vẫn cần runtime smoke. Các ProxyArray .torch fields/body-COM getters đã đối chiếu source nhưng cũng chưa được chạy qua adapter mới.

Angle quantiles dùng sparse histogram resolution **0,1°**, trả conservative upper bin, không giả exact quantile. Nếu upper-bin p95<20° thì không làm gate dễ hơn; giá trị sát20° có thể bị phân loại bảo thủ. Tracking pooled là time/sample-weighted trên first episodes; quantiles RMSE theo env có equal-env weight và giữ duration. Cảnh báo SHORT_TRACKING_EXPOSURE khi first episode<2s; không ghi locomotion PASS từ RMSE thấp.

Raw contact points không được tự bật. Foot net Fz không chứng minh support polygon hợp lệ; không gán NO_CONTACT từ margin NaN. Pure classifier phân biệt missing,invalid,no contact,<3points,degenerate,valid hull và COM outside valid hull; CoP availability độc lập số đỉnh. Normal orientation/contact freshness/point buffers cần thí nghiệm riêng. Production root-cache fix không tự xác nhận contact-data lifecycle.

### B4. Bộ nhớ và output bounds

Runtime defaults64env,20s,chunk16steps; CLI chặn env>256,seconds>60,chunk>256. Conservative chunk budget **N×chunk_steps×320×4bytes**: mặc định **1.310.720bytes=1,25MiB**. Guard cho buffer+stack/compression dùng3×estimate≤max_chunk_mib(default64MiB); mỗi frame được kiểm actual nbytes không vượt schema budget.

Online stats O(N×29) và histogram tối đa3×1801bins/active episode. Chỉ giữ first records của N env và N active accumulators; không giữ list của mọi later episode. Worst-case histogram-entry bound xấp xỉ **6×N×1801**, heap overhead phụ thuộc Python, không nằm trong chunk MiB. Summarizer streaming JSONL, line cap1MiB, chỉ giữ first cohort, later counts/exposure. Không lưu full contact tensors.

Raw float telemetry upper budget cho64×1000steps là81.920.000bytes≈78,125MiB/condition trước nén;30conditions≈2,29GiB, chưa tính JSONL/metadata. Đây là bound từ fields/dtypes, **không phải measured RAM/VRAM**. RSL model/Kit/PhysX GPU usage chưa được đo bằng adapter này.

## C. Unit tests và kiểm tra tĩnh

**60 tests PASS, 0 failures, 0 errors, 0 skipped**, pinned Python3.12.13.
[Structured results](unit-test-results.json), [full final log](unit-tests-60-final.log).

Heavy modules torch/numpy/isaaclab/isaacsim/omni/pxr không có trong test process. --help chạy CPU không mở Kit. Syntax dùng ast.parse/compile(source,...), không ghi bytecode vào files cũ. Cả CLI đã audit; [evaluator help](evaluate_multiskill_c1-help.txt), [summarizer help](summarize_multiskill_c1-help.txt).

| Yêu cầu tối thiểu | Test oracle |
|---|---|
| Fall<2s | S2=0, failure time có giá trị |
| Fall2…5s | S2=1,S5=0 |
| Fall>10s | S10=1,S20=0 |
| Timeout20s | S20=1,physical failure None,KM NOT_REACHED |
| Alive at horizon | S20=1,horizon-censored,không timeout |
| Censor trước mốc đánh giá | S(t)=None; unknown count và bounds được giữ |
| Missing/inconsistent flags | Raise,không đổi thành ongoing/censor |
| Nonfinite data | Reject observation hoặc metric |
| Incomplete matrix | Raise với missing/unexpected list |
| Valid30synthetic conditions | Đủ3models×3scenarios×3seeds +3zero |
| <3contact points | INSUFFICIENT_POINTS,không gọi mất contact |
| Margin NaN | None cho undefined state; valid hull+NaN bị reject |
| No contact | NO_CONTACT với fresh valid empty-load data |
| Tracking rất ngắn | Exposure và warning hiện rõ,không tự PASS |
| Bổ sung | Mixed flags,ties,float tolerance,event-censor tie,checksum/path traversal,output overwrite,LFS pointer,missing snapshots,non-IID labeling,signal separation |

Fixtures chỉ nằm trong synthetic_tests/<unique-id>/ và có **SYNTHETIC_TEST_DATA** marker/dataset_kind/joint labels. Không tạo fake checkpoint binaries. Genuine Stage5 checkpoint chỉ được đọc để test path/header/snapshot handling. Synthetic summary không là evaluation result nghiên cứu; summarizer reject synthetic mặc định, cần --allow-synthetic chỉ cho tests, và reject real/synthetic mix.

Output validator yêu cầu đủ mandatory metrics, đủ first cohort, unique ordering, dt/sample/duration nhất quán, per-env exposure phủ evaluation horizon, valid termination/censor schema và source checksums. Real conditions còn cần initial-state/chunks/provenance. Không bỏ censored records hoặc missing conditions để ghi PASS.

## D. Timeout và censoring

| Biên | Phân loại |
|---|---|
| Physical failure đúng2/5/10/20s | FAIL survival tại đúng mốc và các mốc sau |
| time_out-only đúng20s | Time-limit censor; đã sống đủ20s |
| Physical +time_out cùng bước | Physical failure ưu tiên;timeout flag vẫn được giữ |
| Không done tới cuối horizon | Evaluation-horizon censor,không fabricate failure time |
| Censor trước t | Chưa biết survival tại t;fraction None,bounds rõ |
| Thiếu flag hoặc aggregate sai | Invalid input/runtime error,không silent PASS |

EPS1e−8s xử lý floating representation cho dt0,02; không thêm một control step để làm gate dễ hơn. Median/p10 observed durations không được trình bày như physical failure quantiles. KM giả định non-informative censoring; nếu chưa reached trong followup,không extrapolate.

Summarizer mới không hardcode13conditions. Default yêu cầu đúng30conditions; các seed là eval scene clusters,không training replications. Nó không ghi locomotion PASS; trạng thái complete offline-recomputed luôn kèm **RUNTIME_VERIFICATION_REQUIRED** và support gate NOT_QUALIFIED. Initial fingerprint mismatch thành PAIRING_UNVERIFIED,không giả định seed giống là paired.

## E. Backup manifest và persistence

[backup_manifest.json](backup_manifest.json) là inventory SHA-256 thật,**không phải independent backup**. Manifest bao gồm source provenance/tracked content,hai historical checkpoints,bốn Stage5 checkpoints và snapshots,Stage6 results/logs,hai báo cáo MốcB/C,MốcB14analysis files,official asset,new evaluator/summarizer/tests và readiness evidence. Có per-file bytes,SHA-256,type,priority,Git/LFS evidence và independent-copy status; total/deduplicated bytes là trường authoritative trong manifest.

GitHub/LFS: Stage9 giữ offline,không query remote hoặc upload. Local refs/remotes/origin/g1-multiskill trỏ expected HEAD; các tracked blobs/LFS pointers có provenance từ clone đã có. Hai historical binaries khớp OID/size trong HEAD và local LFS cache. **Local remote-tracking ref/cache không chứng minh live remote object availability hoặc independent backup hiện tại.** Manifest phân biệt REMOTE_SNAPSHOT_EVIDENCE_LIVE_UNVERIFIED với LOCAL_WITH_NO_VERIFIED_INDEPENDENT_COPY. Không khẳng định chắc chắn “chỉ tồn tại ở Vast.ai” nếu chưa kiểm tất cả nguồn ngoài; kết luận là chưa có bản sao độc lập được xác minh.

Các checkpoint/YAML Stage5,Stage6 outputs,báo cáo mới và source Stage9 chưa có bằng chứng off-instance copy. Đây là nhóm ưu tiênP0/P1 cần bảo vệ. Ngoài MultiSkill,inventory đọc cả tracked legacy/FastSAC artifacts để giữ provenance;không chạy hoặc thay FastSAC. Installed venv/wheel caches không được gọi là backed up;manifest ghi package inventories và dependency commits/diff hashes,không giả full environment archive.

Thiếu hoàn toàn: original params/agent.yaml,params/env.yaml,smoke_verification.json của run19/10;18raw MốcB directories. Các C1 checkpoints/configs/evaluation thật chưa được tạo là future artifacts,không phải “backup mất”. Các LFS stubs legacy được đánh pointer-only,binary chưa materialize,không tính133-bytepointer như checkpoint đầy đủ.

Destination options cần người dùng chọn/duyệt:object storage do người dùng quản lý,SFTP/NAS ngoài instance,hoặc repository/artifact store có hỗ trợ kích thước dữ liệu và retention phù hợp. Không tự chọn provider,tạo account hoặc upload. Sau khi được duyệt:copy các fileP0/P1,đọc/download independent copy,verify SHA-256+size;ETag không mặc định SHA-256. Giữ local cho tới khi checksum/restore test PASS;không xóa lịch sử.

## F. Blockers và giới hạn còn lại

- **Runtime integration UNVERIFIED:** chưa xác nhận mới trên PhysX cho .torch shapes,COM timestamps,command hooks,chunks,projected incoming getter hoặc stream performance.
- **Contact quality UNVERIFIED:** raw contact hull không được thu;support margin không primary gate.
- **Backup NOT SATISFIED:** inventory nằm overlay;destination và independent-copy verification chưa có.
- **Budget NOT SATISFIED:** hourly rate/wall-cost cap chưa được duyệt.
- **Training NOT GRANTED:** Stage9 không cho phép C1 hoặc GPU smoke.
- **Historical reproducibility BLOCKED:** thiếu original YAMLs/raw;không ghép YAML mới cho model199.
- C1 matrix implementation được test bằng synthetic30conditions;chưa có real30-condition dataset hay actual standing metrics mới.

Sau approval riêng,runtime smoke nên dùng genuine Stage5 model1/snapshots với1env,4s,seed42,standing,chunk nhỏ;giữ output mới,không học PPO. Mục tiêu chỉ là kiểm adapter API/schema/terminal timing và exact input hashes,không chứng minh standing hoặc whole C1 readiness. Đây là mô tả bước tương lai,không thực thi trong Stage9.

## G. Integrity và file changes

Tạo đúng ba source/test files mới và artifacts/logs/fixtures trong run/stage9-pretraining/. Không sửa bất kỳ file pre-Stage9 nào. Hash baseline gồm173tracked files và64pre-existing outputs/assets;dependency checkout content và inventories270MultiSkill/55main được đối chiếu trước–sau. Git HEAD/branch không đổi;FastSAC giữ nguyên.

Không install package,uv sync,commit/push/reset/clean,thay USD hoặc Vast instance state. GPU không có simulator/compute process cuối lượt. Không có C1 training directory. [Completion audit](completion-audit.json) ghi kết quả đối chiếu sau cùng;file này không tự chứng minh runtime.

Các attempt logs trước được giữ lại;final canonical suite là unit-tests-60-final.log. Source đọc và CLI/schema audit có static_audit.json;test/fixture provenance explicit. Backup manifest loại chính nó khỏi self-hash và không gọi synthetic fixtures là research data;completion audit phát sinh sau manifest nên được ghi là companion evidence,không ngụy tạo recursive checksum.

## H. Một bước tiếp theo đề xuất

**Chốt và phê duyệt một destination backup độc lập với instance cho toàn bộ artifacts ưu tiên, kèm SHA-256 verification/restore procedure.**

Không tự upload hoặc thực hiện bước này. Runtime smoke và C1 training vẫn cần approval riêng sau khi các gate liên quan được xử lý.

