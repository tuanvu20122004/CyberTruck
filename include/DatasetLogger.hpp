#pragma once

#include <array>
#include <fstream>
#include <mutex>
#include <string>

struct DatasetSample {
    long long timestamp_ms;
    int frame_id;
    int is_valid;

    float lateral_deviation;
    float yaw_angle;

    float curvature_0;
    float curvature_1;
    float curvature_2;
    float curvature_3;

    float velocity;
    float prev_steering;
    float expert_steering;
    float steering_sent;

    int servo_command;
    float lane_width_px;
};

class DatasetLogger {
public:
    explicit DatasetLogger(const std::string& filename);
    ~DatasetLogger();

    void logSample(const DatasetSample& sample);
    bool isOpen() const;

private:
    std::ofstream file_;
    mutable std::mutex mutex_;
};