#include "DatasetLogger.hpp"
#include <iostream>

DatasetLogger::DatasetLogger(const std::string& filename) {
    file_.open(filename, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        std::cerr << "[DATASET] Cannot open file: " << filename << std::endl;
        return;
    }

    file_ << "timestamp_ms,frame_id,is_valid,"
          << "lateral_deviation,yaw_angle,"
          << "curvature_0,curvature_1,curvature_2,curvature_3,"
          << "velocity,prev_steering,expert_steering,"
          << "servo_command,lane_width_px\n";
}

DatasetLogger::~DatasetLogger() {
    if (file_.is_open()) {
        file_.close();
    }
}

bool DatasetLogger::isOpen() const {
    return file_.is_open();
}

void DatasetLogger::logSample(const DatasetSample& s) {
    if (!file_.is_open()) return;

    std::lock_guard<std::mutex> lock(mutex_);
    file_ << s.timestamp_ms << ','
          << s.frame_id << ','
          << s.is_valid << ','
          << s.lateral_deviation << ','
          << s.yaw_angle << ','
          << s.curvature_0 << ','
          << s.curvature_1 << ','
          << s.curvature_2 << ','
          << s.curvature_3 << ','
          << s.velocity << ','
          << s.prev_steering << ','
          << s.expert_steering << ','
          << s.servo_command << ','
          << s.lane_width_px
          << '\n';
}