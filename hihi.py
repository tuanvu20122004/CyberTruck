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

MODEL_PATH = r"C:\Users\Admin\Downloads\yolov8n.onnx"

INPUT_SIZE = 640
CONF_THRESHOLD = 0.15
NMS_THRESHOLD = 0.7

FOCAL_LENGTH = 250

CAR_REAL_HEIGHT = 0.22   # kiểm tra lại giá trị này nếu khoảng cách bị sai
SMOOTH_SIZE = 5

UDP_RECV_IP = "0.0.0.0"
YOLO_PORT = 9996         # nhận frame YOLO từ Pi
DEBUG_PORT = 9997        # nhận bird-eye debug từ Pi
UDP_SEND_PORT = 8888     # gửi khoảng cách về Pi
BUFF_SIZE = 65536

RECEIVE_TIMEOUT = 1.0
NO_DATA_EXIT_SECONDS = 10

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

CAR_CLASS_ID = 2  # COCO: car

# ==============================
# SHARED DATA
# ==============================

latest_yolo_view = None
latest_debug_view = None
view_lock = threading.Lock()

running = True
last_pi_ip = None

# chỉ làm mượt cho xe đang chọn
car_pixel_buffer = deque(maxlen=SMOOTH_SIZE)

# ==============================
# DISTANCE
# ==============================

def estimate_distance(focal_length, real_height, pixel_height):
    if pixel_height <= 0:
        return None
    return abs((focal_length * real_height) / pixel_height)

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

        # YOLOv8 ONNX COCO thường ra [84, N] -> transpose thành [N, 84]
        if preds.shape[0] == 84 and preds.shape[1] > 84:
            preds = preds.T

        boxes = []
        scores = []
        class_ids = []

        for det in preds:
            cx, cy, w, h = det[:4]
            class_scores = det[4:]

            cls_id = int(np.argmax(class_scores))
            score = float(class_scores[cls_id])

            # chỉ lấy car
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
    global latest_yolo_view, last_pi_ip, running

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

        best_distance = None
        best_score = -1.0
        best_bbox = None

        # chọn xe có score cao nhất
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            pixel_height = y2 - y1

            if pixel_height < 20:
                continue

            if r["score"] > best_score:
                best_score = r["score"]
                best_bbox = (x1, y1, x2, y2)

        if best_bbox is not None:
            x1, y1, x2, y2 = best_bbox
            pixel_height = y2 - y1

            car_pixel_buffer.append(pixel_height)
            smooth_pixel_height = int(np.median(car_pixel_buffer))

            best_distance = estimate_distance(
                FOCAL_LENGTH,
                CAR_REAL_HEIGHT,
                smooth_pixel_height
            )

            if best_distance is not None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"car {best_distance:.2f} m | {best_score:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # gửi khoảng cách về Pi
        if last_pi_ip is not None:
            message = f"{best_distance:.2f}" if best_distance is not None else "-1.00"

            try:
                send_sock.sendto(message.encode("utf-8"), (last_pi_ip, UDP_SEND_PORT))
            except Exception as e:
                print(f"[YOLO] Loi send distance: {e}")

            cv2.putText(
                frame,
                f"Send to Pi: {message}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
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
    print("✅ Ket thuc chuong trinh.")

if __name__ == "__main__":
    main()
