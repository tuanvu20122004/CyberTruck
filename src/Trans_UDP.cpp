#include "Trans_UDP.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

Trans_UDP::Trans_UDP(const std::string& server_ip, int send_port, int recv_port)
    : server_ip_(server_ip),
      send_port_(send_port),
      recv_port_(recv_port),
      send_sock_(-1),
      recv_sock_(-1),
      distance_(-1.0f)
{
    initSockets();
}

Trans_UDP::~Trans_UDP() {
    closeSockets();
}

bool Trans_UDP::initSockets() {
    send_sock_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (send_sock_ < 0) {
        std::cerr << "[UDP] Khong tao duoc send socket\n";
        return false;
    }

    std::memset(&server_addr_, 0, sizeof(server_addr_));
    server_addr_.sin_family = AF_INET;
    server_addr_.sin_port = htons(send_port_);
    inet_pton(AF_INET, server_ip_.c_str(), &server_addr_.sin_addr);

    std::cout << "[UDP] Send socket -> " << server_ip_ << ":" << send_port_ << std::endl;

    if (recv_port_ > 0) {
        recv_sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (recv_sock_ < 0) {
            std::cerr << "[UDP] Khong tao duoc recv socket\n";
            return false;
        }

        std::memset(&recv_addr_, 0, sizeof(recv_addr_));
        recv_addr_.sin_family = AF_INET;
        recv_addr_.sin_addr.s_addr = INADDR_ANY;
        recv_addr_.sin_port = htons(recv_port_);

        if (bind(recv_sock_, reinterpret_cast<sockaddr*>(&recv_addr_), sizeof(recv_addr_)) < 0) {
            std::cerr << "[UDP] Bind recv socket that bai tai port " << recv_port_ << std::endl;
            return false;
        }

        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 1000; // 1 ms timeout
        setsockopt(recv_sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        std::cout << "[UDP] Recv socket bind tai 0.0.0.0:" << recv_port_ << std::endl;
    }

    return true;
}

void Trans_UDP::sendFrame(const cv::Mat& frame, int quality) {
    if (send_sock_ < 0 || frame.empty()) return;

    std::vector<uchar> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    cv::imencode(".jpg", frame, buf, params);

    sendto(send_sock_, buf.data(), buf.size(), 0,
           reinterpret_cast<sockaddr*>(&server_addr_), sizeof(server_addr_));
}

bool Trans_UDP::receiveDistance() {
    if (recv_sock_ < 0) return false;

    char buffer[128] = {0};
    sockaddr_in sender_addr{};
    socklen_t sender_len = sizeof(sender_addr);

    int n = recvfrom(recv_sock_, buffer, sizeof(buffer) - 1, 0,
                     reinterpret_cast<sockaddr*>(&sender_addr), &sender_len);

    if (n <= 0) {
        return false;
    }

    buffer[n] = '\0';

    try {
        float new_distance = std::stof(buffer);
        distance_ = new_distance;
        // std::cout << "[UDP] Received raw='" << buffer
        //           << "' parsed=" << distance_ << std::endl;
        return true;
    } catch (...) {
        std::cerr << "[UDP] Parse distance fail: '" << buffer << "'" << std::endl;
        return false;
    }
}

float Trans_UDP::getDistance() const {
    return distance_;
}

void Trans_UDP::closeSockets() {
    if (send_sock_ >= 0) close(send_sock_);
    if (recv_sock_ >= 0) close(recv_sock_);
    send_sock_ = -1;
    recv_sock_ = -1;
}