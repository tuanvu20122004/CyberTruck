#ifndef TRANS_UDP_HPP
#define TRANS_UDP_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <arpa/inet.h>
#include <unistd.h>

class Trans_UDP {
public:
    Trans_UDP(const std::string& server_ip, int send_port, int recv_port = -1);
    ~Trans_UDP();

    bool initSockets();
    void sendFrame(const cv::Mat& frame, int quality = 80);

    bool receiveDistance();
    float getDistance() const;

    void closeSockets();

private:
    std::string server_ip_;
    int send_port_;
    int recv_port_;

    int send_sock_;
    int recv_sock_;

    sockaddr_in server_addr_;
    sockaddr_in recv_addr_;

    float distance_;
};

#endif