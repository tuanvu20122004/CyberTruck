import math
import cv2
import numpy as np
import socket
import time
import threading
import onnxruntime as ort
from collections import deque

# ==============================
# CONFIG
# ==============================

MODEL_PATH = r"C:\Users\Admin\Downloads\best.onnx"

INPUT_SIZE = 640
CONF_THRESHOLD = 0.15
NMS_THRESHOLD = 0.7 # tăng NMS để giảm bớt các bbox thừa, tránh nhầm lẫn khi có nhiều xe gần nhau

FOCAL_LENGTH = 250
CAR_REAL_HEIGHT = 0.22

SMOOTH_SIZE = 5

UDP_RECV_IP = "0.0.0.0"
YOLO_PORT = 9996
DEBUG_PORT = 9997
UDP_SEND_PORT = 8888
BUFF_SIZE = 65536

RECEIVE_TIMEOUT = 1.0
NO_DATA_EXIT_SECONDS = 10

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

CAR_CLASS_ID = 0   # custom model: Red_Car

# ==============================
# EGO-LANE FILTER CONFIG
# ==============================

USE_EGO_LANE_FILTER = True

# Chỉ xét vật thể có đáy bbox đủ thấp trong ảnh
MIN_BOTTOM_Y_RATIO = 0.42

# Hành lang ego-lane dạng hình thang quanh tâm ảnh
LANE_HALF_WIDTH_BOTTOM_RATIO = 0.16
LANE_HALF_WIDTH_TOP_RATIO = 0.003

# Giữ object cũ rất ngắn để chống flicker khi vừa ra khỏi lane
OUT_OF_LANE_HOLD_FRAMES = 2

# ==============================
# SHARED DATA
# ==============================

latest_yolo_view = None
latest_debug_view = None
view_lock = threading.Lock()

running = True
last_pi_ip = None

car_pixel_buffer = deque(maxlen=SMOOTH_SIZE)
out_of_lane_counter = 0
fps_value = 0.0
prev_frame_time = None
# ==============================
# DISTANCE
# ==============================

def estimate_distance(focal_length, real_height, pixel_height):
    if pixel_height <= 0:
        return None
    return abs((focal_length * real_height) / pixel_height)


class FrontDistanceFilter:
    def __init__(
        self,
        alpha_near=0.50,     # vật cản gần hơn -> phản ứng nhanh hơn
        alpha_far=0.22,      # vật cản xa hơn -> mượt hơn
        max_jump=1.2,        # reject nếu raw distance nhảy quá lớn trong 1 frame
        hold_frames=3,       # giữ giá trị cũ khi mất detect ngắn hạn
        min_valid=0.05,
        max_valid=10.0
    ):
        self.alpha_near = alpha_near
        self.alpha_far = alpha_far
        self.max_jump = max_jump
        self.hold_frames = hold_frames
        self.min_valid = min_valid
        self.max_valid = max_valid

        self.initialized = False
        self.filtered = None
        self.last_raw = None
        self.miss_count = 0

    def reset(self):
        self.initialized = False
        self.filtered = None
        self.last_raw = None
        self.miss_count = 0

    def update(self, raw_distance, valid=True):
        if (
            (not valid) or
            (raw_distance is None) or
            (not np.isfinite(raw_distance)) or
            (raw_distance < self.min_valid) or
            (raw_distance > self.max_valid)
        ):
            if self.initialized and self.miss_count < self.hold_frames:
                self.miss_count += 1
                return self.filtered
            else:
                self.miss_count += 1
                return None

        raw_distance = float(raw_distance)

        if not self.initialized:
            self.filtered = raw_distance
            self.last_raw = raw_distance
            self.initialized = True
            self.miss_count = 0
            return self.filtered

        if self.last_raw is not None:
            if abs(raw_distance - self.last_raw) > self.max_jump:
                if self.miss_count < self.hold_frames:
                    self.miss_count += 1
                    return self.filtered
                return None

        self.miss_count = 0

        if raw_distance < self.filtered:
            alpha = self.alpha_near
        else:
            alpha = self.alpha_far

        self.filtered = alpha * raw_distance + (1.0 - alpha) * self.filtered
        self.last_raw = raw_distance

        return self.filtered


distance_filter = FrontDistanceFilter()

# ==============================
# EGO-LANE FILTER HELPERS
# ==============================

def lane_half_width_px(y, img_h, img_w):
    """
    Ego-lane corridor dạng hình thang:
    càng gần đáy ảnh thì càng rộng, càng xa thì càng hẹp.
    """
    y = max(0, min(int(y), img_h - 1))
    t = y / float(max(img_h - 1, 1))

    top_half = LANE_HALF_WIDTH_TOP_RATIO * img_w
    bottom_half = LANE_HALF_WIDTH_BOTTOM_RATIO * img_w

    return (1.0 - t) * top_half + t * bottom_half


def is_in_ego_lane(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox

    foot_x = 0.5 * (x1 + x2)
    foot_y = y2

    # vật quá xa / quá cao trong ảnh -> bỏ qua
    if foot_y < MIN_BOTTOM_Y_RATIO * img_h:
        return False

    lane_center_x = 0.5 * img_w
    half_width = lane_half_width_px(foot_y, img_h, img_w)

    return abs(foot_x - lane_center_x) <= half_width

# đoạn này vẽ hành lang ego-lane để debug, có thể bật lên nếu muốn
def draw_ego_lane_corridor(frame):
    h, w = frame.shape[:2]

    y_top = int(MIN_BOTTOM_Y_RATIO * h)
    y_bottom = h - 1
    cx = w // 2

    half_top = int(lane_half_width_px(y_top, h, w))
    half_bottom = int(lane_half_width_px(y_bottom, h, w))

    pts = np.array([
        [cx - half_top, y_top],
        [cx + half_top, y_top],
        [cx + half_bottom, y_bottom],
        [cx - half_bottom, y_bottom]
    ], dtype=np.int32)

    #cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)


# ==============================
# YOLOv8 ONNX
# ==============================

class YOLOv8ONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image):
        h, w = image.shape[:2]

        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (nw, nh))
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)

        pad_x = (INPUT_SIZE - nw) // 2
        pad_y = (INPUT_SIZE - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img, scale, pad_x, pad_y

    def postprocess(self, output, original_shape, scale, pad_x, pad_y):
        h0, w0 = original_shape[:2]

        preds = output
        if len(preds.shape) == 3:
            preds = preds[0]

        # Với YOLOv8 ONNX custom model, output thường là [C, N]
        # Ví dụ 1 class -> [5, 8400], cần transpose thành [8400, 5]
        if len(preds.shape) == 2 and preds.shape[0] < preds.shape[1]:
            preds = preds.T

        boxes = []
        scores = []
        class_ids = []

        for det in preds:
            cx, cy, w, h = det[:4]
            class_scores = det[4:]

            cls_id = int(np.argmax(class_scores))
            score = float(class_scores[cls_id])

            if cls_id != CAR_CLASS_ID:
                continue

            if score < CONF_THRESHOLD:
                continue

            x1 = (cx - w / 2 - pad_x) / scale
            y1 = (cy - h / 2 - pad_y) / scale
            x2 = (cx + w / 2 - pad_x) / scale
            y2 = (cy + h / 2 - pad_y) / scale

            x1 = max(0, min(int(x1), w0 - 1))
            y1 = max(0, min(int(y1), h0 - 1))
            x2 = max(0, min(int(x2), w0 - 1))
            y2 = max(0, min(int(y2), h0 - 1))

            bw = x2 - x1
            bh = y2 - y1

            if bw < 2 or bh < 2:
                continue

            boxes.append([x1, y1, bw, bh])
            scores.append(score)
            class_ids.append(cls_id)

        results = []
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    results.append({
                        "bbox": [x, y, x + w, y + h],
                        "class_id": class_ids[i],
                        "score": scores[i]
                    })

        return results

    def detect(self, image):
        inp, scale, pad_x, pad_y = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: inp})
        return self.postprocess(outputs[0], image.shape, scale, pad_x, pad_y)

# ==============================
# UDP HELPERS
# ==============================

def make_sock(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
    s.bind((UDP_RECV_IP, port))
    s.settimeout(RECEIVE_TIMEOUT)
    return s


def recv_frame(sock):
    packet, addr = sock.recvfrom(BUFF_SIZE)
    frame = cv2.imdecode(np.frombuffer(packet, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame, addr

# ==============================
# YOLO THREAD
# ==============================

def yolo_thread():
    global latest_yolo_view, last_pi_ip, running, out_of_lane_counter, fps_value, prev_frame_time

    detector = YOLOv8ONNX(MODEL_PATH)

    recv_sock = make_sock(YOLO_PORT)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"[YOLO] Dang cho frame tai {UDP_RECV_IP}:{YOLO_PORT}")

    last_frame_time = time.time()

    while running:
        try:
            frame, addr = recv_frame(recv_sock)
            last_frame_time = time.time()
            last_pi_ip = addr[0]
            current_time = time.time()
            if prev_frame_time is not None:
                instant_fps = 1.0 / max(current_time - prev_frame_time, 1e-6)
                if fps_value == 0.0:
                    fps_value = instant_fps
                else:
                    fps_value = 0.9 * fps_value + 0.1 * instant_fps
            prev_frame_time = current_time
        except socket.timeout:
            if time.time() - last_frame_time > NO_DATA_EXIT_SECONDS:
                print("[YOLO] Khong nhan duoc du lieu qua lau.")
                continue
            continue
        except Exception as e:
            print(f"[YOLO] Loi recv: {e}")
            continue

        if frame is None:
            continue

        results = detector.detect(frame)

        h, w = frame.shape[:2]

        if USE_EGO_LANE_FILTER:
            draw_ego_lane_corridor(frame)

        raw_distance = None
        filtered_distance = None
        best_score = -1.0
        best_bbox = None
        best_height = -1

        # Chỉ chọn xe nằm trong ego lane
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            bbox = (x1, y1, x2, y2)

            pixel_height = y2 - y1
            if pixel_height < 20:
                continue

            in_lane = True
            if USE_EGO_LANE_FILTER:
                in_lane = is_in_ego_lane(bbox, w, h)

            color = (0, 255, 0) if in_lane else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            foot_x = int(0.5 * (x1 + x2))
            foot_y = int(y2)
            cv2.circle(frame, (foot_x, foot_y), 4, color, -1)

            cv2.putText(
                frame,
                f"{'IN' if in_lane else 'OUT'} lane",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            if not in_lane:
                continue

            # Trong lane thì ưu tiên bbox cao nhất
            if pixel_height > best_height:
                best_height = pixel_height
                best_score = r["score"]
                best_bbox = bbox

        if best_bbox is not None:
            out_of_lane_counter = 0

            x1, y1, x2, y2 = best_bbox
            pixel_height = y2 - y1

            # median filter cho pixel height
            car_pixel_buffer.append(pixel_height)
            smooth_pixel_height = int(np.median(car_pixel_buffer))

            raw_distance = estimate_distance(
                FOCAL_LENGTH,
                CAR_REAL_HEIGHT,
                smooth_pixel_height
            )

            filtered_distance = distance_filter.update(raw_distance, valid=True)

            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            if filtered_distance is not None:
                label = (
                    f"car raw:{raw_distance:.2f}m "
                    f"filt:{filtered_distance:.2f}m "
                    f"| conf:{best_score:.2f}"
                )
            else:
                label = f"car raw:{raw_distance:.2f}m filt:None | conf:{best_score:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 30, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            send_distance = filtered_distance
        else:
            filtered_distance = distance_filter.update(None, valid=False)
            out_of_lane_counter += 1

            # Chỉ hold rất ngắn để chống flicker, tránh giữ xe lane cũ quá lâu
            if out_of_lane_counter <= OUT_OF_LANE_HOLD_FRAMES and filtered_distance is not None:
                send_distance = filtered_distance
            else:
                send_distance = None
                car_pixel_buffer.clear()
                distance_filter.reset()

            cv2.putText(
                frame,
                f"No IN-LANE car -> send: "
                f"{'None' if send_distance is None else f'{send_distance:.2f}m'}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # Gửi khoảng cách đã lọc về Pi
        if last_pi_ip is not None:
            message = f"{send_distance:.2f}" if send_distance is not None else "-1.00"

            try:
                send_sock.sendto(message.encode("utf-8"), (last_pi_ip, UDP_SEND_PORT))
            except Exception as e:
                print(f"[YOLO] Loi send distance: {e}")

            #cv2.putText(
            #    frame,
            #    f"Send to Pi: {message}",
            #    (10, 30),
            #    cv2.FONT_HERSHEY_SIMPLEX,
            #    0.8,
            #    (0, 0, 255),
            #    2
            #)
            cv2.putText(
                frame,
                f"FPS: {fps_value:.2f}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        with view_lock:
            latest_yolo_view = display_frame.copy()

    recv_sock.close()
    send_sock.close()

# ==============================
# DEBUG THREAD
# ==============================

def debug_thread():
    global latest_debug_view, running

    recv_sock = make_sock(DEBUG_PORT)
    print(f"[DEBUG] Dang cho frame tai {UDP_RECV_IP}:{DEBUG_PORT}")

    last_frame_time = time.time()

    while running:
        try:
            frame, addr = recv_frame(recv_sock)
            last_frame_time = time.time()
        except socket.timeout:
            if time.time() - last_frame_time > NO_DATA_EXIT_SECONDS:
                continue
            continue
        except Exception as e:
            print(f"[DEBUG] Loi recv: {e}")
            continue

        if frame is None:
            continue

        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        with view_lock:
            latest_debug_view = display_frame.copy()

    recv_sock.close()

# ==============================
# MAIN
# ==============================

def main():
    global running

    t1 = threading.Thread(target=yolo_thread, daemon=True)
    t2 = threading.Thread(target=debug_thread, daemon=True)

    t1.start()
    t2.start()

    print("[MAIN] Da khoi dong 2 luong:")
    print(f"       - YOLO  port {YOLO_PORT}")
    print(f"       - DEBUG port {DEBUG_PORT}")

    while True:
        yolo_view = None
        debug_view = None

        with view_lock:
            if latest_yolo_view is not None:
                yolo_view = latest_yolo_view.copy()
            if latest_debug_view is not None:
                debug_view = latest_debug_view.copy()

        if yolo_view is not None:
            cv2.imshow("YOLO Input + Car Distance", yolo_view)

        if debug_view is not None:
            cv2.imshow("BirdEye Debug", debug_view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            running = False
            break

        time.sleep(0.001)

    cv2.destroyAllWindows()
    print("Ket thuc chuong trinh.")


if __name__ == "__main__":
    main()