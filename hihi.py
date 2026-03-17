import sys, time, threading, serial
from collections import deque
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ===== Cấu hình =====
PORT        = "COM3"       # đổi theo cổng STM32
BAUD        = 115200
TIMEOUT     = 1

HZ          = 100          # tần số gửi lệnh
PERIOD      = 1.0 / HZ

ANGLE_MIN   = 40           # yêu cầu mới: 40
ANGLE_MAX   = 140          # yêu cầu mới: 140
ANGLE_STEP  = 10

PATTERN     = "ramp"       # "ramp" = 40→140 rồi nhảy về 40; "triangle" = 40→140→40

# ===== UART =====
try:
    ser = serial.Serial(PORT, baudrate=BAUD, timeout=TIMEOUT)
    print(f"✅ UART opened on {PORT} (baud={BAUD})")
except Exception as e:
    print(f"❌ Không mở được cổng {PORT}: {e}")
    sys.exit(1)

# ===== Log CSV =====
csv = open("feedback_log.csv", "w", encoding="utf-8")
csv.write("time,setpoint,speed,error_percent\n")
csv.flush()

# ===== Trạng thái dùng chung =====
running = True
start_time = time.time()

# Dữ liệu vẽ (giới hạn để không phình RAM)
MAX_POINTS = 500
times     = deque(maxlen=MAX_POINTS)
setpoints = deque(maxlen=MAX_POINTS)
speeds    = deque(maxlen=MAX_POINTS)

def parse_feedback(line: str):
    """
    Format kỳ vọng:  FB,SET,<float>,SPD,<float>
    Trả về (setpoint, speed) hoặc None nếu sai format.
    """
    parts = line.split(",")
    if len(parts) == 5 and parts[0] == "FB" and parts[1] == "SET" and parts[3] == "SPD":
        try:
            sp  = float(parts[2])
            spd = float(parts[4])
            return sp, spd
        except ValueError:
            return None
    return None

def read_feedback():
    global running
    while running:
        try:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            if not line:
                continue

            parsed = parse_feedback(line)
            if parsed is None:
                continue

            sp, v = parsed
            t = time.time() - start_time

            # cập nhật dữ liệu đồ thị
            times.append(t)
            setpoints.append(sp)
            speeds.append(v)

            # ghi CSV + sai số %
            err = 0.0 if sp == 0 else abs((v - sp) / sp) * 100.0
            csv.write(f"{t:.3f},{sp:.3f},{v:.3f},{err:.2f}\n")
            csv.flush()

        except Exception as e:
            print("❌ Lỗi đọc UART:", e)
            break

def send_commands():
    global running
    # nhập setpoint 1 lần
    try:
        spd_input = input("Nhập setpoint tốc độ (m/s) (Enter = 0.0): ")
        spd = float(spd_input) if spd_input.strip() else 0.0
    except Exception:
        spd = 0.0

    # tạo dãy góc theo PATTERN
    angles_up = list(range(ANGLE_MIN, ANGLE_MAX + 1, ANGLE_STEP))
    if PATTERN == "triangle":
        angles_down = angles_up[-2::-1]  # bỏ đỉnh để tránh lặp
        seq = angles_up + angles_down
        note = f"{ANGLE_MIN}↔{ANGLE_MAX}"
    else:
        seq = angles_up
        note = f"{ANGLE_MIN}→{ANGLE_MAX}"

    print(f"🚀 Gửi tự động góc {note} @ {HZ}Hz, setpoint={spd:.2f}")
    cycler = it.cycle(seq)

    next_send = time.perf_counter()
    last_print = time.time()

    while running:
        ang = next(cycler)
        cmd = f"CMD,{spd:.2f},{ang}\r\n"
        try:
            ser.write(cmd.encode())
        except Exception as e:
            print("❌ Lỗi gửi UART:", e)
            break

        # log TX ~mỗi 1s
        now = time.time()
        if now - last_print >= 1.0:
            print(f"[TX] {cmd.strip()}")
            last_print = now

        # nhịp gửi chính xác theo HZ
        next_send += PERIOD
        sleep_time = next_send - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_send = time.perf_counter()

# ===== Vẽ realtime =====
fig, ax = plt.subplots()
(line_speed,) = ax.plot([], [], label="Speed (m/s)")
(line_sp,)    = ax.plot([], [], label="Setpoint (m/s)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity (m/s)")
ax.grid(True)
ax.legend(loc="upper left")

def update(_):
    line_speed.set_data(list(times), list(speeds))
    line_sp.set_data(list(times), list(setpoints))
    ax.relim()
    ax.autoscale_view()
    return line_speed, line_sp

ani = animation.FuncAnimation(
    fig, update, interval=200, blit=False, cache_frame_data=False  # tránh UserWarning
)

# ===== Main =====
reader = threading.Thread(target=read_feedback, daemon=True)
sender = threading.Thread(target=send_commands, daemon=True)
reader.start()
sender.start()

try:
    plt.show()
except KeyboardInterrupt:
    print("\n👋 Dừng bởi người dùng")
finally:
    running = False
    try:
        sender.join(timeout=1.0)
        reader.join(timeout=1.0)
    except:
        pass
    try:
        ser.close()
    except:
        pass
    try:
        csv.close()
    except:
        pass
    print("🔌 UART closed. File feedback_log.csv đã được lưu.")
