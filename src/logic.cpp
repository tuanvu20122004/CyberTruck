#include "logic.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <pthread.h>
#include <cmath>
#include <fstream>
#include <iomanip>

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

const char* controlModeToString(ControlMode m)
{
    switch (m)
    {
    case ControlMode::MPC:           return "MPC";
    case ControlMode::PURE_PURSUIT:  return "PP";
    default:                         return "UNKNOWN";
    }
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

Logic::Logic(const std::string& videoPath)
    : detector(videoPath, 640, 480),
      comm("/dev/ttyACM0", 115200),
      udp_yolo("192.168.1.101", 9996, 8888),
      udp_debug("192.168.1.101", 9997)
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
    pure_pursuit.setLookahead(0.35f);          // tune
    pure_pursuit.setRearAxleOffsetPx(40.0f);   // tune
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

    // Chọn controller mặc định
    control_mode = ControlMode::PURE_PURSUIT;

    if (!detector.isOpened())
    {
        std::cerr << "[LOGIC] Camera not opened." << std::endl;
        throw std::runtime_error("LaneDetector not opened");
    }
}

float Logic::computeSteering(
    const std::vector<cv::Point>& base_centerline,
    const std::vector<cv::Point>& target_centerline,
    const cv::Mat& birdEyeView,
    float distance,
    float& lateral_error_out,
    float& yaw_out)
{
    lateral_error_out = 0.0f;
    yaw_out = 0.0f;

    const float trigger_distance = 1.1f;
    const bool use_base_path = (distance < 0.0f || distance > trigger_distance);

    // =============================
    // MPC branch
    // =============================
    if (control_mode == ControlMode::MPC)
    {
        MpcState state_base = mpc.computeMpcParameters(base_centerline, birdEyeView);
        MpcState state_target = mpc.computeMpcParameters(target_centerline, birdEyeView);

        if (!state_base.is_valid || !state_target.is_valid)
            return 0.0f;

        if (use_base_path)
        {
            lateral_error_out = state_base.lateral_deviation;
            yaw_out = state_base.yaw_angle;
            return mpc.computeSteeringAngle(state_base, desired_velocity);
        }
        else
        {
            lateral_error_out = state_target.lateral_deviation;
            yaw_out = state_target.yaw_angle;
            return mpc.computeSteeringAngle(state_target, desired_velocity);
        }
    }

    // =============================
    // Pure Pursuit branch
    // =============================
    float meter_per_pixel = planner.getMeterPerPixel();
    if (meter_per_pixel > 1e-6f)
    {
        pure_pursuit.setPixelPerMeter(1.0f / meter_per_pixel);
    }

    const std::vector<cv::Point>& path = use_base_path ? base_centerline : target_centerline;

    // PP không trả ey/yaw như MPC, nên để 0 để giữ format log
    lateral_error_out = 0.0f;
    yaw_out = 0.0f;

    return pure_pursuit.computeSteeringAngle(
        path,
        birdEyeView.size(),
        desired_velocity
    );
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
                 << ",control_mode"
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

            // ESC / q -> quit
            if (key == 27 || key == 'q' || key == 'Q')
            {
                running.store(false);
                break;
            }

            // m -> switch MPC
            if (key == 'm' || key == 'M')
            {
                control_mode = ControlMode::MPC;
                std::cout << "[CTRL] Switch to MPC" << std::endl;
            }

            // p -> switch Pure Pursuit
            if (key == 'p' || key == 'P')
            {
                control_mode = ControlMode::PURE_PURSUIT;
                std::cout << "[CTRL] Switch to Pure Pursuit" << std::endl;
            }
        }
    });

    std::thread control_thread([&]()
    {
        bindToCore(1);

        cv::Mat frame_local;
        auto last_send = std::chrono::steady_clock::now();
        auto last_log_time = std::chrono::steady_clock::now();

        static float last_valid_distance = -1.0f;
        static auto last_distance_time = std::chrono::steady_clock::now();

        PlannerState prev_state = PlannerState::KEEP_LANE;

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
            auto now_distance = std::chrono::steady_clock::now();
            auto distance_age_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now_distance - last_distance_time
                ).count();

            if (distance_age_ms < obstacle_timeout_ms)
            {
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
                350, // hoặc detector.getLaneWidthPx()
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

                // Nonlinear steering map - giữ nguyên để so sánh công bằng
                steering = 1.5f * std::pow(steering, 3) + 2.0f * steering;

                if (steering <= -25.0f) steering = -25.0f;
                else if (steering >= 25.0f) steering = 25.0f;

                int servo = static_cast<int>(std::lround(97.0f + steering));

                float velocity_cmd = desired_velocity;
                comm.sendCommands(velocity_cmd, servo);

                if (planner.getState() != prev_state && log_file.is_open())
                {
                    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - run_start).count();

                    log_file << "# EVENT state_change"
                             << " t=" << elapsed_ms
                             << " ctrl=" << controlModeToString(control_mode)
                             << " new_state=" << plannerStateToString(planner.getState())
                             << " dir=" << plannerDirToString(planner.getLastDirection())
                             << std::endl;

                    prev_state = planner.getState();
                }

                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time).count() >= 100)
                {
                    last_log_time = now;

                    if (log_file.is_open())
                    {
                        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now - run_start).count();

                        log_file << elapsed_ms << ","
                                 << controlModeToString(control_mode) << ","
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
                        << "ctrl=" << controlModeToString(control_mode)
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

    if (camera_thread.joinable()) camera_thread.join();
    if (control_thread.joinable()) control_thread.join();

    if (log_file.is_open())
        log_file.close();

    std::cout << "[LOGIC] Stopped cleanly." << std::endl;
}