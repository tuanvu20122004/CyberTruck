// logic.cpp
#include "logic.hpp"

#include <iostream>
#include <thread>
#include <chrono>
#include <pthread.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

namespace
{
const char* plannerStateToString(PlannerState s)
{
    switch (s)
    {
    case PlannerState::KEEP_LANE:           return "KEEP";
    case PlannerState::CHANGE_USING_DASHED: return "CHANGE";
    case PlannerState::FOLLOW_LANE:         return "FOLLOW";
    default:                                return "UNKNOWN";
    }
}

const char* plannerDirToString(Type_Change_t d)
{
    switch (d)
    {
    case CHANGE_LEFT:  return "LEFT";
    case CHANGE_RIGHT: return "RIGHT";
    case DONT_CHANGE:
    default:           return "NONE";
    }
}

void bindToCore(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();

    int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

    if (result != 0)
        std::cerr << "[LOGIC] Error setting thread affinity." << std::endl;
    else
        std::cout << "[LOGIC] Thread bound to core " << core_id << std::endl;
}

float saturate(float value, float min_value, float max_value)
{
    return std::max(min_value, std::min(value, max_value));
}

} // namespace

const char* Logic::controlModeToString(ControlMode mode)
{
    switch (mode)
    {
    case ControlMode::MPC:          return "MPC";
    case ControlMode::PURE_PURSUIT: return "PURE_PURSUIT";
    default:                        return "UNKNOWN";
    }
}

Logic::Logic(const std::string& videoPath)
    : detector(videoPath, 640, 480),
      comm("/dev/ttyACM0", 115200),
      udp_yolo("192.168.1.114", 9996, 8888),
      udp_debug("192.168.1.114", 9997)
{
    // =========================
    // MPC config
    // =========================
    mpc.init(1000.0f, 50.0f, 5.0f);
    mpc.debugMatrices();
    mpc.setVehicleParams(0.2515f, 2.3f, 0.132f, 0.12f, 0.04f, 0.02f, 0.04f);

    // =========================
    // Pure Pursuit config
    // =========================
    pure_pursuit.setWheelbase(0.2515f);
    pure_pursuit.setLookahead(0.35f);
    pure_pursuit.setRearAxleOffsetPx(40.0f);
    pure_pursuit.setMaxSteeringDeg(25.0f);

    // =========================
    // Planner config
    // =========================
    planner.setLaneWidthMeters(0.40f);
    planner.setVehicleSize(0.21f, 0.432f);
    planner.setObstacleSize(0.20f, 0.22f);
    planner.setSafeMargin(0.08f);
    planner.setSpeed(desired_velocity);
    planner.setTriggerDistance(1.1f);

    if (!detector.isOpened())
    {
        std::cerr << "[LOGIC] Camera not opened." << std::endl;
        throw std::runtime_error("LaneDetector not opened");
    }

    std::cout << "[LOGIC] Default controller = "
              << controlModeToString(control_mode.load()) << std::endl;
}

float Logic::computeSteering(
    const std::vector<cv::Point>& base_centerline,
    const std::vector<cv::Point>& target_centerline,
    const cv::Mat& birdEyeView,
    float distance,
    float& lateral_error_out,
    float& yaw_out
)
{
    lateral_error_out = 0.0f;
    yaw_out = 0.0f;

    if (birdEyeView.empty())
        return 0.0f;

    // Nếu obstacle còn xa thì bám base_centerline như logic cũ.
    // Nếu obstacle gần thì dùng target_centerline do planner sinh ra.
    const bool use_base_path =
        (distance < 0.0f) || (distance > 1.1f) || (target_centerline.size() < 3);

    const std::vector<cv::Point>& active_path =
        use_base_path ? base_centerline : target_centerline;

    if (active_path.size() < 2)
        return 0.0f;

    // Luôn dùng MPC state để fill log lateral/yaw
    MpcState active_state = mpc.computeMpcParameters(active_path, birdEyeView);
    lateral_error_out = active_state.lateral_deviation;
    yaw_out = active_state.yaw_angle;

    float steering_deg = 0.0f;
    ControlMode mode = control_mode.load();

    if (mode == ControlMode::MPC)
    {
        if (!active_state.is_valid)
            return 0.0f;

        steering_deg = mpc.computeSteeringAngle(active_state, desired_velocity);
    }
    else
    {
        const float meter_per_pixel = planner.getMeterPerPixel();
        if (meter_per_pixel > 1e-6f)
        {
            pure_pursuit.setPixelPerMeter(1.0f / meter_per_pixel);
        }
        else
        {
            // fallback theo lane width nếu planner scale chưa ổn
            const float lane_width_px = detector.getLaneWidthPx();
            if (lane_width_px > 1e-3f)
                pure_pursuit.setPixelPerMeter(lane_width_px / 0.40f);
        }

        steering_deg = pure_pursuit.computeSteeringAngle(
            active_path,
            birdEyeView.size(),
            desired_velocity
        );
    }

    // Giữ nguyên phần map steering -> servo như code cũ
    steering_deg = 1.5f * std::pow(steering_deg, 3) + 10.0f * steering_deg;
    steering_deg = saturate(steering_deg, -25.0f, 25.0f);

    return steering_deg;
}

void Logic::run()
{
    std::ofstream log_file("planner_log.txt", std::ios::out | std::ios::trunc);
    if (!log_file.is_open())
    {
        std::cerr << "[LOGIC] Cannot open planner_log.txt" << std::endl;
    }
    else
    {
        log_file << "time_ms"
                 << ",state"
                 << ",direction"
                 << ",obs_distance_m"
                 << ",min_distance_m"
                 << ",min_ttc_s"
                 << ",cost"
                 << ",lateral_error_m"
                 << ",yaw_rad"
                 << ",steering_deg"
                 << ",servo"
                 << ",lane_width_px"
                 << ",meter_per_pixel"
                 << std::endl;
    }

    auto run_start = std::chrono::steady_clock::now();

    std::thread camera_thread([&]()
    {
        bindToCore(0);

        cv::Mat frame;
        auto last_yolo_send = std::chrono::steady_clock::now();

        while (running.load())
        {
            if (!detector.getFrame(frame))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            cv::Mat frame_yolo;

            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                latest_frame = detector.getFrameResize().clone();
                frame_yolo = latest_frame.clone();
            }

            auto now = std::chrono::steady_clock::now();
            if (!frame_yolo.empty() &&
                std::chrono::duration_cast<std::chrono::milliseconds>(now - last_yolo_send).count() >= 50)
            {
                last_yolo_send = now;
                udp_yolo.sendFrame(frame_yolo, 85);
            }

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q')
            {
                running.store(false);
                break;
            }
            else if (key == 'm' || key == 'M')
            {
                control_mode.store(ControlMode::MPC);
                std::cout << "[LOGIC] Switched controller -> MPC" << std::endl;
            }
            else if (key == 'p' || key == 'P')
            {
                control_mode.store(ControlMode::PURE_PURSUIT);
                std::cout << "[LOGIC] Switched controller -> PURE_PURSUIT" << std::endl;
            }
        }
    });

    std::thread control_thread([&]()
    {
        bindToCore(1);

        cv::Mat frame_local;
        auto last_send = std::chrono::steady_clock::now();
        auto last_log_time = std::chrono::steady_clock::now();

        float last_valid_distance = -1.0f;
        auto last_distance_time = std::chrono::steady_clock::now();

        PlannerState prev_state = PlannerState::KEEP_LANE;
        ControlMode prev_mode = control_mode.load();

        while (running.load())
        {
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                if (!latest_frame.empty())
                    frame_local = latest_frame.clone();
                else
                    frame_local.release();
            }

            if (frame_local.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            detector.processFrame(frame_local);

            std::vector<cv::Point> base_centerline = detector.getCenterline();
            cv::Mat birdEyeView = detector.getBirdEyeView();

            if (udp_yolo.receiveDistance())
            {
                float d = udp_yolo.getDistance();
                if (d > 0.05f && d < 5.0f)
                {
                    last_valid_distance = d;
                    last_distance_time = std::chrono::steady_clock::now();
                }
            }

            float distance = -1.0f;
            {
                auto now_distance = std::chrono::steady_clock::now();
                auto distance_age_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        now_distance - last_distance_time
                    ).count();

                if (distance_age_ms < obstacle_timeout_ms)
                    distance = last_valid_distance;
            }

            int planner_width  = !birdEyeView.empty() ? birdEyeView.cols : frame_local.cols;
            int planner_height = !birdEyeView.empty() ? birdEyeView.rows : frame_local.rows;

            planner.setSpeed(desired_velocity);

            cv::Mat bev = detector.getBirdEyeView();
            if (!bev.empty())
                udp_debug.sendFrame(bev, 80);

            std::vector<cv::Point> target_centerline = planner.update(
                base_centerline,
                detector.getLeftCoeffs(),
                detector.getRightCoeffs(),
                detector.hasLeftLane(),
                detector.hasRightLane(),
                detector.getLaneWidthPx(),
                detector.left_type,
                detector.right_type,
                distance,
                planner_width,
                planner_height
            );

            if (target_centerline.size() < 3)
                target_centerline = base_centerline;

            auto now = std::chrono::steady_clock::now();

            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send).count() >= 50)
            {
                last_send = now;

                float lateral_error = 0.0f;
                float yaw = 0.0f;
                float steering = computeSteering(
                    base_centerline,
                    target_centerline,
                    birdEyeView,
                    distance,
                    lateral_error,
                    yaw
                );

                int servo = static_cast<int>(std::lround(static_cast<float>(servo_center) + steering));
                servo = std::max(0, std::min(180, servo));

                float velocity_cmd = desired_velocity;
                comm.sendCommands(velocity_cmd, servo);

                if (planner.getState() != prev_state && log_file.is_open())
                {
                    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - run_start).count();

                    log_file << "# EVENT state_change"
                             << " t=" << elapsed_ms
                             << " new_state=" << plannerStateToString(planner.getState())
                             << " dir=" << plannerDirToString(planner.getLastDirection())
                             << std::endl;

                    prev_state = planner.getState();
                }

                ControlMode current_mode = control_mode.load();
                if (current_mode != prev_mode && log_file.is_open())
                {
                    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - run_start).count();

                    log_file << "# EVENT control_mode_change"
                             << " t=" << elapsed_ms
                             << " new_mode=" << controlModeToString(current_mode)
                             << std::endl;

                    prev_mode = current_mode;
                }

                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time).count() >= 100)
                {
                    last_log_time = now;

                    if (log_file.is_open())
                    {
                        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now - run_start).count();

                        log_file << elapsed_ms << ","
                                 << plannerStateToString(planner.getState()) << ","
                                 << plannerDirToString(planner.getLastDirection()) << ","
                                 << std::fixed << std::setprecision(3)
                                 << distance << ","
                                 << planner.getLastMinDistance() << ","
                                 << planner.getLastMinTTC() << ","
                                 << planner.getLastCost() << ","
                                 << lateral_error << ","
                                 << yaw << ","
                                 << steering << ","
                                 << servo << ","
                                 << detector.getLaneWidthPx() << ","
                                 << planner.getMeterPerPixel()
                                 << std::endl;
                    }

                    std::cout
                        << "[RUN] "
                        << "mode=" << controlModeToString(current_mode)
                        << " state=" << plannerStateToString(planner.getState())
                        << " dir=" << plannerDirToString(planner.getLastDirection())
                        << " obs=" << distance
                        << " minD=" << planner.getLastMinDistance()
                        << " minTTC=" << planner.getLastMinTTC()
                        << " cost=" << planner.getLastCost()
                        << " ey=" << lateral_error
                        << " yaw=" << yaw
                        << " steer=" << steering
                        << " servo=" << servo
                        << std::endl;
                }
            }
        }
    });

    while (running.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    if (camera_thread.joinable())  camera_thread.join();
    if (control_thread.joinable()) control_thread.join();

    if (log_file.is_open())
        log_file.close();

    std::cout << "[LOGIC] Stopped cleanly." << std::endl;
}