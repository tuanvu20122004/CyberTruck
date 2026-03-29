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

float clampf(float v, float lo, float hi)
{
    return std::max(lo, std::min(v, hi));
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
    // Planner config
    // =========================
    planner.setLaneWidthMeters(0.40f);
    planner.setVehicleSize(0.21f, 0.432f);
    planner.setObstacleSize(0.20f, 0.22f);
    planner.setSafeMargin(0.08f);
    planner.setSpeed(desired_velocity);
    planner.setTriggerDistance(1.1f);

    // =========================
    // Pure Pursuit config
    // =========================
    // Luu y: can khai bao pure_pursuit trong logic.hpp
    // vi du: PurePursuitController pure_pursuit;
    pure_pursuit.setWheelbase(0.2515f);
    pure_pursuit.setLookahead(0.35f);
    pure_pursuit.setRearAxleOffsetPx(40.0f);
    pure_pursuit.setMaxSteeringDeg(25.0f);

    if (!detector.isOpened())
    {
        std::cerr << "[LOGIC] Camera not opened." << std::endl;
        throw std::runtime_error("LaneDetector not opened");
    }
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
        }
    });

    std::thread pp_thread([&]()
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
                    now_distance - last_distance_time).count();

            if (distance_age_ms < obstacle_timeout_ms)
                distance = last_valid_distance;

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
                350, // co the doi lai detector.getLaneWidthPx() sau khi he thong on dinh
                detector.left_type,
                detector.right_type,
                distance,
                planner_width,
                planner_height);

            if (target_centerline.size() < 3)
                target_centerline = base_centerline;

            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send).count() >= 50)
            {
                last_send = now;

                if (!birdEyeView.empty() && target_centerline.size() >= 2)
                {
                    float meter_per_pixel = planner.getMeterPerPixel();
                    if (meter_per_pixel > 1e-6f)
                        pure_pursuit.setPixelPerMeter(1.0f / meter_per_pixel);

                    float steering = 0.0f;

                    // Khi obstacle con xa hon trigger, bam lane goc.
                    // Khi obstacle nam trong trigger, bam quy dao chuyen lan do planner sinh ra.
                    if (distance > 1.1f)
                    {
                        steering = pure_pursuit.computeSteeringAngle(
                            base_centerline,
                            birdEyeView.size(),
                            desired_velocity);
                    }
                    else
                    {
                        steering = pure_pursuit.computeSteeringAngle(
                            target_centerline,
                            birdEyeView.size(),
                            desired_velocity);
                    }

                    // Giu lai nonlinear map cu de de so sanh cong bang voi ban MPC.
                    steering = 1.5f * std::pow(steering, 3) + 2.0f * steering;
                    steering = clampf(steering, -25.0f, 25.0f);

                    int servo = static_cast<int>(std::lround(97.0f + steering));
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
                                     << steering << ","
                                     << servo << ","
                                     << detector.getLaneWidthPx() << ","
                                     << planner.getMeterPerPixel()
                                     << std::endl;
                        }

                        std::cout
                            << "[RUN] "
                            << "state=" << plannerStateToString(planner.getState())
                            << " dir=" << plannerDirToString(planner.getLastDirection())
                            << " obs=" << distance
                            << " minD=" << planner.getLastMinDistance()
                            << " minTTC=" << planner.getLastMinTTC()
                            << " cost=" << planner.getLastCost()
                            << " steer=" << steering
                            << " servo=" << servo
                            << std::endl;
                    }
                }
            }
        }
    });

    while (running.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    if (camera_thread.joinable()) camera_thread.join();
    if (pp_thread.joinable()) pp_thread.join();

    if (log_file.is_open())
        log_file.close();

    std::cout << "[LOGIC] Stopped cleanly." << std::endl;
}
