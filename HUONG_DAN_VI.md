# Hướng dẫn mô phỏng robot Unitree G1

Tài liệu này tổng hợp công việc đã thực hiện, cách tái tạo môi trường bằng `uv`, cách chạy mô phỏng, cách sử dụng checkpoint sau huấn luyện và sự khác nhau giữa Isaac Sim, Isaac Lab, PhysX và Newton.

Thư mục thực tế của dự án là **`/workspace/Humaniod`**. Giữ đúng cách viết và chữ hoa khi nhập lệnh. Tên môi trường là **`Template-G1-Datn-v0`**.

## 1. Công việc đã hoàn thành

1. **Kiểm tra dự án và phần cứng:** xác định GPU RTX 5060 Ti có 16 GB VRAM, desktop ảo tại `DISPLAY=:20`, mã nguồn môi trường G1 và các checkpoint có sẵn.
2. **Sửa cấu hình phụ thuộc:** file `pyproject.toml` ban đầu chọn Windows và trỏ tới một bản Isaac Lab không có trên máy. Đã chuyển sang Linux x86_64, dùng mã nguồn tại `.deps/IsaacLab`, thêm gói dự án `g1_datn` và Isaac Sim, bỏ các nhóm phụ thuộc mẫu không dùng.
3. **Cài đặt bằng uv:** tạo môi trường Python `.venv`, cài dự án ở chế độ editable để thay đổi mã nguồn có hiệu lực khi chạy lại, và cập nhật `uv.lock` để lưu phiên bản các gói.
4. **Giải quyết xung đột phiên bản:** khai báo các override trong cấu hình uv cho những ràng buộc không khớp giữa Isaac Lab và Isaac Sim, gồm PyTorch, Newton, coverage, packaging và NumPy. Giữ nguyên driver NVIDIA do máy chủ cung cấp.
5. **Viết script cài đặt và chạy:** thêm `setup.sh`, `run.sh`, `start-desktop.sh` và `run-desktop.sh` để có thể cài lại và khởi động bằng một lệnh.
6. **Bổ sung kiểm tra mô phỏng:** thêm `--max_steps` vào script zero agent và playback, kiểm tra reward hoặc action không chứa giá trị NaN/vô hạn, và in kết quả hoàn thành. Playback báo tiến độ mỗi 500 bước và kiểm tra khi cửa sổ viewer đóng.
7. **Xử lý hiển thị:** cửa sổ Kit dùng Vulkan bị trống trên desktop Xvfb của máy này. Đã dùng Newton Viewer với OpenGL qua VirtualGL để hiển thị; phần vật lý vẫn chạy bằng PhysX của Isaac Sim trên GPU.
8. **Thiết lập dịch vụ:** tạo dịch vụ supervisor tên `humanoid`, chạy bằng người dùng desktop, ghi log vào hệ thống của instance và tự khởi động lại nếu tiến trình thoát bất thường.
9. **Kiểm chứng:** chạy mô phỏng vật lý 100 bước với bốn robot, chạy checkpoint 500 bước không mở cửa sổ, chạy demo có cửa sổ hơn 5.000 bước, quan sát robot chuyển động và chạy lại thành công lệnh cài đặt một lần.

Checkpoint đi bộ đã có sẵn trong dự án. Công việc trên là cài đặt, cấu hình và chạy kiểm tra checkpoint đó. Chưa thực hiện một đợt huấn luyện mới hoặc đánh giá đầy đủ chất lượng policy.

## 2. Các thành phần khác nhau như thế nào?

| Thành phần | Vai trò | Cách dùng trong dự án này |
| --- | --- | --- |
| **Isaac Sim** | Nền tảng mô phỏng robot, tích hợp vật lý, dựng hình, cảm biến mô phỏng và công cụ làm việc với thế giới ảo. | Cung cấp môi trường chạy cho backend PhysX. |
| **PhysX** | Bộ máy tính toán vật lý: chuyển động, lực, tiếp xúc và va chạm. | Tính chuyển động của robot trên CUDA. |
| **Isaac Lab** | Framework xây dựng môi trường học cho robot: observation, action, reward, reset và tích hợp thuật toán học. | Định nghĩa môi trường G1 và kết nối mô phỏng với quá trình huấn luyện/playback. |
| **Newton physics engine** | Bộ máy mô phỏng vật lý dựa trên NVIDIA Warp, có thể sử dụng GPU. | Không được chọn làm backend vật lý trong cấu hình demo hiện tại. |
| **Newton Viewer** | Công cụ hiển thị tương tác; ở đây dùng OpenGL để vẽ robot và môi trường. | Hiển thị trạng thái do PhysX tính toán. |
| **RSL-RL** | Thư viện thuật toán reinforcement learning. | Huấn luyện hoặc nạp policy đã lưu để tạo action điều khiển robot. |

**Newton Viewer và Newton physics engine có vai trò khác nhau.** Cửa sổ có tên “Newton Viewer” không có nghĩa là vật lý đang chạy bằng Newton.

Luồng hoạt động của dự án:

```text
Checkpoint chứa trọng số policy đã học
                  ↓
Policy nhận observation và tạo action điều khiển khớp
                  ↓
Isaac Sim / PhysX tính chuyển động, tiếp xúc và va chạm
                  ↓
Isaac Lab cập nhật observation, reward và điều kiện reset
                  ↓
Newton Viewer hiển thị trạng thái robot
                  ↓
Lặp lại ở bước mô phỏng tiếp theo
```

Thay viewer chỉ đổi cách hiển thị. Chuyển backend vật lý từ PhysX sang Newton là thay đổi riêng, cần kiểm tra lại hành vi của policy vì kết quả mô phỏng có thể khác.

## 3. Môi trường đã cài

| Thành phần | Phiên bản hoặc cấu hình |
| --- | --- |
| Hệ điều hành mục tiêu | Linux x86_64 |
| Python | 3.12 |
| Isaac Lab | Tag `v3.0.0-beta2` |
| Commit Isaac Lab | `28a37cecdd433c22d9eabd6a5954add9f13a8951` |
| Isaac Sim | `6.0.0.1` |
| PyTorch | `2.10.0+cu128` |
| CUDA của bản PyTorch | 12.8 |
| RSL-RL | `5.0.1` |
| GPU đã kiểm tra | RTX 5060 Ti, 16 GB VRAM |
| Số robot của demo mặc định | 4 |
| Backend vật lý | PhysX trên `cuda:0` |
| Viewer | Newton OpenGL qua VirtualGL |

Lần cài đầu cần Internet để tải gói và tài nguyên robot. Nên dành khoảng 60 GB dung lượng trống cho môi trường và cache. Launcher đặt `OMNI_KIT_ACCEPT_EULA=YES` để Isaac Sim khởi động không cần nhập xác nhận giấy phép trong terminal.

## 4. Cài đặt và khởi động bằng một lệnh

Từ bản dự án đã có đủ mã nguồn, script, lockfile và checkpoint:

```bash
cd /workspace/Humaniod
bash setup.sh --desktop
```

Script sẽ:

1. Cài uv nếu chưa có.
2. Tải Isaac Lab vào `.deps/IsaacLab` nếu chưa có và kiểm tra đúng commit.
3. Chạy `uv sync --frozen --python 3.12` để tái tạo môi trường từ `uv.lock`.
4. Chạy 100 bước mô phỏng với bốn robot bằng zero agent để kiểm tra GPU.
5. Tạo hoặc cập nhật cấu hình supervisor rồi khởi động demo.

Nếu chỉ muốn cài và kiểm tra GPU:

```bash
bash setup.sh
```

Tùy chọn `--desktop` được viết cho image Vast Linux desktop hiện tại, có supervisor, sudo, các script hỗ trợ và display `:20`. Nó không tự cài một hệ thống desktop lên máy Linux bất kỳ.

## 5. Chạy mô phỏng sau khi đã cài xong

Mở **Selkies Low Latency Desktop** từ portal của Vast, rồi chạy:

```bash
cd /workspace/Humaniod
./start-desktop.sh
```

Tìm cửa sổ **Newton Viewer** để xem robot. Viewer dùng cơ chế đăng nhập sẵn có của desktop, không cần mở thêm cổng công khai.

Các lệnh quản lý:

```bash
# Xem trạng thái dịch vụ
sudo supervisorctl status humanoid

# Xem log trực tiếp; Ctrl+C chỉ dừng xem log
tail -f /var/log/portal/humanoid.log

# Dừng demo
sudo supervisorctl stop humanoid

# Chạy lại dịch vụ đã được tạo
sudo supervisorctl start humanoid

# Khởi động lại sau khi thay đổi mã nguồn hoặc checkpoint
sudo supervisorctl restart humanoid
```

Để chạy trực tiếp trong terminal của desktop, trước tiên dừng dịch vụ để tránh chạy hai bản demo:

```bash
sudo supervisorctl stop humanoid
cd /workspace/Humaniod
./run.sh
```

Khi không nhận tham số, `run.sh` tự chọn môi trường Python, VirtualGL nếu có, bốn robot, Newton Viewer và checkpoint mặc định. Có thể đóng viewer hoặc nhấn Ctrl+C trong terminal chạy trực tiếp để kết thúc.

Lần khởi động đầu có thể mất thời gian tải tài nguyên và khởi tạo viewer. Trạng thái `RUNNING` của supervisor chỉ xác nhận tiến trình đang chạy; chưa chắc cửa sổ đã tải xong.

## 6. Sau khi huấn luyện xong, chạy checkpoint mới thế nào?

**`start-desktop.sh` không tự chọn checkpoint mới nhất.** Nó chạy `run-desktop.sh`, sau đó gọi `run.sh`. Đường dẫn checkpoint mặc định hiện nằm trong `run.sh`:

```text
logs/rsl_rl/g1_datn/2026-09-17_03-30-23/best_walk.pt
```

Checkpoint là file `.pt` lưu trạng thái mô hình; khi playback, policy dùng trọng số đã học để tạo action. Chạy demo không tiếp tục huấn luyện.

Để sử dụng kết quả huấn luyện mới:

1. Xác định file `.pt` muốn chạy trong thư mục kết quả huấn luyện.
2. Mở `/workspace/Humaniod/run.sh` và thay đường dẫn sau `--checkpoint` bằng đường dẫn file mới. Giữ nguyên các tham số còn lại nếu vẫn dùng cùng môi trường và cấu hình policy.
3. Nếu dịch vụ đang chạy, thực hiện:

   ```bash
   sudo supervisorctl restart humanoid
   ```

4. Nếu dịch vụ đang dừng hoặc chưa được tạo, thực hiện:

   ```bash
   cd /workspace/Humaniod
   ./start-desktop.sh
   ```

5. Xem log và cửa sổ Newton Viewer để xác nhận checkpoint mới được nạp.

Chỉ chạy lại `start-desktop.sh` khi dịch vụ đang chạy sẽ không buộc tiến trình nạp lại checkpoint. Sau khi đổi checkpoint, dùng lệnh **restart**.

Checkpoint cần tương thích với cấu hình môi trường và kiến trúc policy hiện tại, đặc biệt là kích thước observation và action. Nếu huấn luyện với cấu hình khác, phải dùng cấu hình tương ứng khi playback.

## 7. Huấn luyện và kiểm tra không mở cửa sổ

Ví dụ lệnh bắt đầu huấn luyện:

```bash
cd /workspace/Humaniod
./run.sh scripts/rsl_rl/train.py \
  --task Template-G1-Datn-v0 \
  --num_envs 64 \
  --visualizer none
```

Đây là lệnh hướng dẫn; chưa chạy kiểm chứng một đợt huấn luyện hoàn chỉnh. Điều chỉnh `--num_envs` theo dung lượng GPU. Có thể dừng demo trước khi huấn luyện để dành tài nguyên GPU.

Kiểm tra checkpoint có sẵn trong 500 bước, không mở cửa sổ:

```bash
./run.sh scripts/rsl_rl/play.py \
  --task Template-G1-Datn-v0 \
  --num_envs 4 \
  --visualizer none \
  --max_steps 500 \
  --checkpoint logs/rsl_rl/g1_datn/2026-09-17_03-30-23/best_walk.pt
```

Thay đường dẫn cuối bằng checkpoint mới nếu cần. `--max_steps 0` không đặt giới hạn số bước; khi chạy không có viewer, dùng Ctrl+C để dừng. Khi có viewer, đóng cửa sổ cũng kết thúc playback.

Kiểm tra vật lý với action bằng 0:

```bash
./run.sh scripts/zero_agent.py \
  --task Template-G1-Datn-v0 \
  --num_envs 4 \
  --visualizer none \
  --max_steps 100
```

Zero agent dùng để xác nhận mô phỏng chạy được, không phải policy đi bộ. Trong demo dùng policy, robot vẫn có thể ngã và được reset theo điều kiện của môi trường.

## 8. Chức năng từng file

| File hoặc thư mục | Chức năng |
| --- | --- |
| `setup.sh` | Cài môi trường bằng uv, kiểm tra GPU và tùy chọn chạy dịch vụ desktop. |
| `pyproject.toml` | Khai báo dependency, đường dẫn mã nguồn editable, index tải gói và override phiên bản. |
| `uv.lock` | Lưu các phiên bản gói đã giải quyết để tái tạo môi trường. |
| `.venv/` | Môi trường Python của dự án. |
| `.deps/IsaacLab/` | Mã nguồn Isaac Lab tại commit được cố định. |
| `run.sh` | Thiết lập biến môi trường, chọn Python và chạy policy mặc định hoặc script được truyền vào. |
| `start-desktop.sh` | Tạo/cập nhật dịch vụ supervisor `humanoid` và khởi động nếu cần. |
| `run-desktop.sh` | Kết nối launcher với display `:20` và hệ thống log của instance. |
| `scripts/zero_agent.py` | Chạy thử môi trường với action bằng 0 và số bước giới hạn. |
| `scripts/rsl_rl/play.py` | Nạp checkpoint và chạy policy trong mô phỏng. |
| `scripts/rsl_rl/train.py` | Điểm vào để huấn luyện bằng RSL-RL. |
| `logs/rsl_rl/g1_datn/` | Các kết quả huấn luyện và checkpoint của dự án. |
| `.gitignore` | Loại khỏi danh sách theo dõi Git các thư mục dependency, môi trường, file phát sinh và trạng thái viewer. |
| `SETUP.md` | Ghi chép cài đặt và vận hành bằng tiếng Anh. |
| `HUONG_DAN_VI.md` | Tài liệu tổng hợp bằng tiếng Việt này. |

Cấu hình dịch vụ trên hệ thống: `/etc/supervisor/conf.d/humanoid.conf`.
Log demo: `/var/log/portal/humanoid.log`.

## 9. Các kiểm tra đã thực hiện

| Kiểm tra | Kết quả ghi nhận |
| --- | --- |
| Zero agent | Bốn G1 chạy 100 bước PhysX trên `cuda:0`, reward hữu hạn. |
| Checkpoint có sẵn | Chạy 500 bước không có cửa sổ, action hữu hạn. |
| Demo trực quan | Chạy hơn 5.000 bước và quan sát được robot chuyển động. |
| Cài lại bằng script | Chạy lại thành công `setup.sh --desktop` với môi trường đã cài. |

Bằng chứng lưu trên máy:

- `run/setup.log`: log kiểm tra cài đặt ban đầu.
- `run/play-check.log`: log chạy thử checkpoint.
- `run/final-setup.log`: log kiểm tra chạy lại script.
- `run/verified-desktop.png`: ảnh chụp demo đã kiểm tra.

Đây là kết quả tại thời điểm hoàn thành thiết lập. Muốn biết dịch vụ hiện còn chạy hay không, dùng `sudo supervisorctl status humanoid`. Các kiểm tra trên chưa chứng minh policy đi bộ tốt trong mọi tình huống hoặc có thể triển khai trực tiếp lên robot thật.

## 10. Xử lý sự cố

- **Không thấy robot:** mở đúng Selkies desktop, tìm Newton Viewer, kiểm tra trạng thái dịch vụ và log. Chờ quá trình tải tài nguyên hoàn tất.
- **Cửa sổ Kit bị trống:** dùng launcher mặc định với `--visualizer newton`; đây là đường hiển thị đã xác nhận hoạt động trên máy này.
- **Vẫn chạy model cũ:** kiểm tra đường dẫn `--checkpoint` trong `run.sh`, sau đó restart dịch vụ.
- **Lỗi kích thước khi nạp checkpoint:** đối chiếu cấu hình observation, action và kiến trúc mạng với cấu hình đã dùng khi huấn luyện.
- **Thiếu bộ nhớ GPU:** dừng các bản demo chạy trùng và giảm `--num_envs` cho lần chạy tiếp theo.
- **Script báo sai revision Isaac Lab:** `.deps/IsaacLab` không ở commit mong đợi. Sao lưu các sửa đổi nếu có trước khi thay bản checkout.
- **Chạy từ thư mục khác:** dùng đường dẫn đầy đủ, ví dụ `bash /workspace/Humaniod/setup.sh --desktop`.
- **Chuyển sang máy khác:** cần GPU/driver tương thích và môi trường Linux phù hợp. Các thành phần desktop trong script phụ thuộc image Vast hiện tại.

Không chạy `uv lock --upgrade` nếu chưa sẵn sàng kiểm tra lại tương thích; lệnh này có thể thay các phiên bản đã được xác nhận hoạt động.

## 11. Sao lưu để tái tạo môi trường

Giữ cùng nhau mã nguồn dự án, checkpoint, `pyproject.toml`, `uv.lock` và bốn script `.sh`. Script cài đặt không tự tải lại checkpoint riêng của bạn nếu checkpoint bị mất. `.venv` và `.deps` có thể được tái tạo từ script.

Trên instance đã kiểm tra, `/workspace` **không nằm trên host volume bền vững**. Dữ liệu có thể mất khi recycle hoặc destroy instance. Sao lưu mã nguồn và kết quả huấn luyện ra ngoài instance trước những thao tác đó.

Khi chuyển thư mục dự án, chạy lại `setup.sh --desktop` để cấu hình supervisor trỏ tới vị trí mới. Tránh dùng đường dẫn có khoảng trắng cho cấu hình dịch vụ hiện tại.

## 12. Tài liệu chính thức

- [Cài đặt Isaac Lab v3.0.0-beta2](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/setup/installation/pip_installation.html)
- [Giới thiệu Isaac Sim](https://developer.nvidia.com/isaac/sim/)
- [Giới thiệu Newton Physics](https://newton-physics.github.io/newton/latest/tutorials/00_introduction.html)
- [Newton Viewer và công cụ quan sát mô phỏng](https://newton-physics.github.io/newton/stable/guide/visualization.html)

Các trang tài liệu mới nhất có thể mô tả phiên bản khác với phiên bản cố định trong dự án; khi tái tạo môi trường này, dùng `setup.sh` và `uv.lock` đi kèm.
