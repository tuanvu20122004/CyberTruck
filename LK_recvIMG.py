import cv2
import numpy as np
import socket
import time

def main(server_ip="0.0.0.0", port=9996):
    BUFF_SIZE = 65536
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
    sock.bind((server_ip, port))
    sock.settimeout(2.0)

    print(f"Đang chờ stream ảnh UDP tại {server_ip}:{port} ...")

    last_frame_time = time.time()
    TIMEOUT_SECONDS = 100
    window_created = False

    while True:
        try:
            packet, _ = sock.recvfrom(BUFF_SIZE)
            last_frame_time = time.time()
        except socket.timeout:
            if time.time() - last_frame_time > TIMEOUT_SECONDS:
                print("Không nhận được dữ liệu quá lâu. Thoát.")
                break
            continue

        npdata = np.frombuffer(packet, dtype=np.uint8)
        frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
        if frame is None:
            print("Decode lỗi.")
            continue

        h, w = frame.shape[:2]
        print(f"Received frame: {w}x{h}")

        if not window_created:
            cv2.namedWindow("UDP Stream", cv2.WINDOW_AUTOSIZE)
            window_created = True

        cv2.imshow("UDP Stream", frame)

        if cv2.waitKey(1) == 27:
            break

    sock.close()
    cv2.destroyAllWindows()
    print("Kết thúc hiển thị video.")

if __name__ == "__main__":
    main("0.0.0.0", 9996)