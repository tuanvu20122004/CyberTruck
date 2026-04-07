#include "logic_shadow_mode.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <sstream>
#include <thread>

namespace {

void bindToCore(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    const int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        std::cerr << "[LOGIC] Error setting thread affinity to core " << core_id << std::endl;
    } else {
        std::cout << "[LOGIC] Thread bound to core " << core_id << std::endl;
    }
}

// nội suy góc đánh lái
float remapSteeringForActuator(float raw_steering)
{
    float sent = 0.02f * std::pow(raw_steering, 3.0f) + 1.15f * raw_steering;
    sent = std::clamp(sent, -25.0f, 25.0f);
    return sent;
}

} // namespace

Logic::Logic(const std::string& videoPath, const std::string& policyPath)
    : detector(videoPath, 640, 480),
      mpc(),
      comm("/dev/ttyACM0", 115200),
      udp_send("192.168.1.100", 9996),
      policy_model(policyPath),
      logger("policy_shadow_log.txt")
{
    mpc.init(1000.0f, 50.0f, 5.0f);
    mpc.setVehicleParams(0.2515f, 2.3f, 0.132f, 0.12f, 0.04f, 0.02f, 0.04f);
    mpc.debugMatrices();

    if (!detector.isOpened()) {
        throw std::runtime_error("[LOGIC] LaneDetector not opened");
    }
    if (!policy_model.isLoaded()) {
        throw std::runtime_error("[LOGIC] Policy model not loaded");
    }

    openShadowCsv("policy_shadow_log.csv");
    std::cout << "[LOGIC] Shadow mode started with policy JSON: " << policyPath << std::endl;
}

void Logic::openShadowCsv(const std::string& filename)
{
    shadow_csv.open(filename, std::ios::out | std::ios::trunc);
    if (!shadow_csv.is_open()) {
        throw std::runtime_error("[LOGIC] Cannot open shadow CSV: " + filename);
    }

    shadow_csv << "timestamp_ms,frame_id,is_valid,"
               << "lateral_deviation,yaw_angle,"
               << "curvature_0,curvature_1,curvature_2,curvature_3,"
               << "velocity,prev_steering_raw,"
               << "raw_steering_mpc,raw_steering_pred,raw_abs_error,"
               << "steering_sent_mpc,servo_command,lane_width_px\n";
}

// ghi dữ liệu từ MPC vào CSV
void Logic::logShadowRow(long long timestamp_ms,
                         int frame_id,
                         const MpcState& state,
                         float prev_raw_steering,
                         float raw_steering_mpc,
                         float raw_steering_pred,
                         float steering_sent_mpc,
                         int servo_command,
                         float lane_width_px)
{
    if (!shadow_csv.is_open()) {
        return;
    }

    shadow_csv << timestamp_ms << ','
               << frame_id << ','
               << 1 << ','
               << state.lateral_deviation << ','
               << state.yaw_angle << ','
               << (state.curvature.size() > 0 ? state.curvature[0] : 0.0f) << ','
               << (state.curvature.size() > 1 ? state.curvature[1] : 0.0f) << ','
               << (state.curvature.size() > 2 ? state.curvature[2] : 0.0f) << ','
               << (state.curvature.size() > 3 ? state.curvature[3] : 0.0f) << ','
               << desired_velocity << ','
               << prev_raw_steering << ','
               << raw_steering_mpc << ','
               << raw_steering_pred << ','
               << std::abs(raw_steering_mpc - raw_steering_pred) << ','
               << steering_sent_mpc << ','
               << servo_command << ','
               << lane_width_px
               << '\n';
}

void Logic::cameraLoop()
{
    bindToCore(0);

    cv::Mat frame;
    while (running.load()) {
        if (!detector.getFrame(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            latest_frame = detector.getFrameResize().clone();
        }

        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            running.store(false);
            break;
        }
    }
}

void Logic::controlLoop()
{
    bindToCore(1);

    cv::Mat frame_local;
    auto last_send = std::chrono::steady_clock::now();
    float prev_raw_steering = 0.0f;

    while (running.load()) {
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (!latest_frame.empty()) {
                frame_local = latest_frame.clone();
            } else {
                frame_local.release();
            }
        }

        if (frame_local.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        detector.processFrame(frame_local);

        const std::vector<cv::Point> centerline = detector.getCenterline();
        const cv::Mat birdEyeView = detector.getBirdEyeView();
        const MpcState state = mpc.computeMpcParameters(centerline, birdEyeView);

        const cv::Mat bev = detector.getBirdEyeView();
        if (!bev.empty()) {
            udp_send.sendFrame(bev, 60);
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }

        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send).count() < 50) {
            continue;
        }
        last_send = now;

        if (!state.is_valid) {
            continue;
        }

        // build đầu vào vào bộ model
        const PolicyModel::FeatureVector features =
            PolicyModel::buildFeatures(state, desired_velocity, prev_raw_steering);

        // tính góc lái dự đoán và góc lái từ bộ mpc
        const float raw_steering_pred = policy_model.infer(features);
        const float raw_steering_mpc = mpc.computeSteeringAngle(state, desired_velocity);

        // Shadow mode: xe van lai bang MPC, policy chi du doan va duoc ghi log.
        const float steering_sent_mpc = remapSteeringForActuator(raw_steering_mpc);
        const int servo_command = static_cast<int>(std::lround(97.0f + steering_sent_mpc));
        comm.sendCommands(desired_velocity, servo_command);

        const long long timestamp_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        const int current_frame_id = frame_id_++;

        logShadowRow(timestamp_ms,
                     current_frame_id,
                     state,
                     prev_raw_steering,
                     raw_steering_mpc,
                     raw_steering_pred,
                     steering_sent_mpc,
                     servo_command,
                     detector.getLaneWidthPx());

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4)
           << "shadow raw_mpc=" << raw_steering_mpc
           << ", raw_pred=" << raw_steering_pred
           << ", abs_err=" << std::abs(raw_steering_mpc - raw_steering_pred)
           << ", ey=" << state.lateral_deviation
           << ", yaw=" << state.yaw_angle;
        logger.log(ss.str(), 0.0);

        std::cout << "[SHADOW] frame=" << current_frame_id
                  << " raw_mpc=" << raw_steering_mpc
                  << " raw_pred=" << raw_steering_pred
                  << " abs_err=" << std::abs(raw_steering_mpc - raw_steering_pred)
                  << std::endl;

        // Prev steering cho feature o che do shadow nen la lenh thuc su da duoc thi hanh.
        prev_raw_steering = raw_steering_mpc;
    }
}

void Logic::run()
{
    std::thread camera_thread(&Logic::cameraLoop, this);
    std::thread control_thread(&Logic::controlLoop, this);

    while (running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (camera_thread.joinable()) {
        camera_thread.join();
    }
    if (control_thread.joinable()) {
        control_thread.join();
    }

    if (shadow_csv.is_open()) {
        shadow_csv.close();
    }

    std::cout << "[LOGIC] Stopped cleanly." << std::endl;
}
