#include "logic.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <pthread.h>
#include <cmath>
#include <fstream>
#include <iomanip>

#include <thread>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

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
    mpc.init(1000.0f, 50.0f, 5.0f);
    mpc.debugMatrices();

    mpc.setVehicleParams(0.2515f, 2.3f, 0.132f, 0.12f, 0.04f, 0.02f, 0.04f);

    // =========================
    // Planner config
    // =========================
    planner.setLaneWidthMeters(0.40f); //Giá trị cần tune lại
    planner.setVehicleSize(0.21f, 0.432f);// Giá trị cần tune lại
    planner.setObstacleSize(0.20f, 0.22f);// Giá trị cần tune lại
    planner.setSafeMargin(0.1f);// Giá trị cần tune lại nếu thấy xe đổi lane sát quá thì tăng thêm, nếu thấy đổi lane quá xa thì giảm bớt đây là khoảng cách an toàn giữa xe mình với obstacle khi đổi lane
    planner.setSpeed(desired_velocity);
    planner.setTriggerDistance(1.1f);

    if (!detector.isOpened())
    {
        std::cerr << "[LOGIC] Camera not opened." << std::endl;
        throw std::runtime_error("LaneDetector not opened");
    }

    //std::cout << "[LOGIC] MPC initialized." << std::endl;
    //std::cout << "[LOGIC] Planner initialized. Vx = " << desired_velocity << " m/s" << std::endl;
}

void Logic::setControlMode(ControlMode mode)
{
    control_mode = mode;

    if (control_mode == ControlMode::MPC)
        std::cout << "[LOGIC] Using MPC controller\n";
    else
        std::cout << "[LOGIC] Using Pure Pursuit controller\n";
}

// hàm đọc phím từ terminal
int getch_nonblock()
{
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;

    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    return ch;
}

// luồng đọc input
void keyboardThread(std::atomic<bool>& running_flag)
{
    while (true)
    {
        int key = getch_nonblock();

        if (key == 'R' || key == 'r')
        {
            running_flag = true;
            std::cout << "[CMD] RUN\n";
        }
        else if (key == 'S' || key == 's')
        {
            running_flag = false;
            std::cout << "[CMD] STOP\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
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
                 << ",lateral_error_m"
                 << ",yaw_rad"
                 << ",steering_deg"
                 << ",servo"
                 << ",lane_width_px"
                 << ",meter_per_pixel"
                 << std::endl;
    }

    auto run_start = std::chrono::steady_clock::now();

    // luồng đọc tín hiệu từ t
    std::thread kb_thread(keyboardThread, std::ref(is_running));

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
                std::chrono::duration_cast<std::chrono::milliseconds>(now - last_yolo_send).count() >= 20)
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

    std::thread mpc_thread([&]()
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
            MpcState state1 = mpc.computeMpcParameters(base_centerline, birdEyeView);
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
            if(!bev.empty())
                udp_debug.sendFrame(bev, 80);

            std::vector<cv::Point> target_centerline = planner.update(
                base_centerline,
                detector.getLeftCoeffs(),
                detector.getRightCoeffs(),
                detector.hasLeftLane(),
                detector.hasRightLane(),
                350,//detector.getLaneWidthPx(),
                detector.left_type,
                detector.right_type,
                distance,
                planner_width,
                planner_height
            );

            if (target_centerline.size() < 3)
                target_centerline = base_centerline;

            MpcState state = mpc.computeMpcParameters(target_centerline, birdEyeView);
            
            auto now = std::chrono::steady_clock::now();

            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send).count() >= 20)
            {
                last_send = now;

                if (state.is_valid)
                {
                    float steering = 0.0f;
                    float velocity_cmd = 0.0f;
                    // Obstacle gate:
                    // - distance = -1.0f nghĩa là không có obstacle hợp lệ
                    // - chỉ dùng target_centerline khi obstacle tồn tại và nằm trong trigger 1.1m
                    const bool has_obstacle = (distance > 0.05f);
                    const bool avoid_obstacle = has_obstacle && (distance <= 1.1f);

                    if(is_running == true)
                    {
                        velocity_cmd = desired_velocity;
                        if (control_mode == ControlMode::MPC)
                        {
                            if (!avoid_obstacle)
                            {
                                steering = mpc.computeSteeringAngle(state1, desired_velocity);
                            }
                            else
                            {
                                steering = mpc.computeSteeringAngle(state, desired_velocity);
                            }
                        }

                        else // PURE_PURSUIT
                        {
                            if (!avoid_obstacle)
                            {
                                steering = pure_pursuit.computeSteeringAngle(
                                    base_centerline,
                                    birdEyeView.size(),
                                    desired_velocity
                                );
                            }
                            else
                            {
                                steering = pure_pursuit.computeSteeringAngle(
                                    target_centerline,
                                    birdEyeView.size(),
                                    desired_velocity
                                );
                            }
                        }
                    }

                    else
                    {
                        // STOP override
                        steering = 0.0f;
                        velocity_cmd = 0.0;
                    }

                    if (control_mode == ControlMode::MPC)
                    {
                        steering = 0.02f * std::pow(steering, 3) + 1.5f * steering;
                    }
                    else
                    {
                        // Pure Pursuit 
                        steering = 0.02f * std::pow(steering, 3) + 1.2f * steering;
                    }

                    if (steering <= -25.0f) steering = -25.0f;
                    else if (steering >= 25.0f) steering = 25.0f;

                    int servo = static_cast<int>(std::lround(94.0f + steering));

                    
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

                    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time).count() >= 50)
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
                                     << state.lateral_deviation << ","
                                     << state.yaw_angle << ","
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
                            << " ey=" << state.lateral_deviation
                            << " yaw=" << state.yaw_angle
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
    if (mpc_thread.joinable()) mpc_thread.join();
    if (kb_thread.joinable()) kb_thread.join();


    if (log_file.is_open())
        log_file.close();

    std::cout << "[LOGIC] Stopped cleanly." << std::endl;
}