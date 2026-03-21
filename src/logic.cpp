#include "logic.hpp"

#include <iostream>
#include <thread>
#include <chrono>
#include <pthread.h>
#include <cmath>
#include <stdexcept>

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
      udp_yolo("192.168.1.115", 9996, 8888),
      udp_debug("192.168.1.115", 9997)
{
    mpc.init(1000.0f, 50.0f, 5.0f);
    mpc.debugMatrices();

    mpc.setVehicleParams(0.2515f, 2.3f, 0.132f, 0.12f, 0.04f, 0.02f, 0.04f);

    if (!detector.isOpened())
    {
        std::cerr << "[LOGIC] Camera not opened." << std::endl;
        throw std::runtime_error("LaneDetector not opened");
    }

    std::cout << "[LOGIC] MPC initialized." << std::endl;
}

void Logic::run()
{
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

    std::thread mpc_thread([&]()
    {
        bindToCore(1);

        cv::Mat frame_local;

        auto last_send = std::chrono::steady_clock::now();
        auto last_debug_send = std::chrono::steady_clock::now();
        auto last_decision_tick = std::chrono::steady_clock::now();

        float last_valid_distance = -1.0f;
        auto last_distance_time = std::chrono::steady_clock::now();

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

            // =========================
            // Receive / hold obstacle distance
            // =========================
            if (udp_yolo.receiveDistance())
            {
                float d = udp_yolo.getDistance();
                if (d > 0.0f)
                {
                    last_valid_distance = d;
                    last_distance_time = std::chrono::steady_clock::now();
                }
            }

            auto now_dist = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now_dist - last_distance_time).count() > 500)
            {
                last_valid_distance = -1.0f;
            }

            float distance = last_valid_distance;

            int planner_width  = !birdEyeView.empty() ? birdEyeView.cols : frame_local.cols;
            int planner_height = !birdEyeView.empty() ? birdEyeView.rows : frame_local.rows;

            auto now_tick = std::chrono::steady_clock::now();
            float dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                           now_tick - last_decision_tick).count() / 1000.0f;
            last_decision_tick = now_tick;

            if (dt < 0.001f)
                dt = 0.05f;

            // =========================
            // Decision layer
            // =========================
            LaneChangeDecision::Input decision_in;
            decision_in.base_centerline = base_centerline;
            decision_in.left_coeff = detector.getLeftCoeffs();
            decision_in.right_coeff = detector.getRightCoeffs();
            decision_in.has_left_lane = detector.hasLeftLane();
            decision_in.has_right_lane = detector.hasRightLane();
            decision_in.left_type = detector.left_type;
            decision_in.right_type = detector.right_type;
            decision_in.lane_width_px = detector.getLaneWidthPx();
            decision_in.obstacle_distance = distance;
            decision_in.ego_speed = desired_velocity;
            decision_in.dt = dt;
            decision_in.img_width = planner_width;
            decision_in.img_height = planner_height;

            LaneChangeDecision::Output decision_out = lane_decision.update(decision_in);

            if (decision_out.approve_lane_change && !planner.isLaneChangeActive())
            {
                if (decision_out.direction == DecisionDirection::LEFT)
                {
                    planner.requestLaneChange(CHANGE_LEFT);
                    lane_decision.notifyLaneChangeStarted();
                    std::cout << "[LOGIC] Decision approve: CHANGE_LEFT" << std::endl;
                }
                else if (decision_out.direction == DecisionDirection::RIGHT)
                {
                    planner.requestLaneChange(CHANGE_RIGHT);
                    lane_decision.notifyLaneChangeStarted();
                    std::cout << "[LOGIC] Decision approve: CHANGE_RIGHT" << std::endl;
                }
            }

            // =========================
            // Planner layer
            // =========================
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

            if (planner.isLaneChangeFinished())
            {
                lane_decision.notifyLaneChangeFinished();
                planner.clearFinishedFlag();
                std::cout << "[LOGIC] Lane change finished -> decision cooldown" << std::endl;
            }

            if (target_centerline.size() < 3)
                target_centerline = base_centerline;

            // =========================
            // Debug view
            // =========================
            auto now_debug = std::chrono::steady_clock::now();

            if (!birdEyeView.empty() &&
                std::chrono::duration_cast<std::chrono::milliseconds>(now_debug - last_debug_send).count() >= 100)
            {
                last_debug_send = now_debug;

                cv::Mat debug_view = birdEyeView.clone();

                Logger::drawPolyline(debug_view, base_centerline, cv::Scalar(255, 0, 0));
                Logger::drawPolyline(debug_view, target_centerline, cv::Scalar(0, 0, 255));

                std::string state_text = "Decision state: " +
                    std::to_string(static_cast<int>(decision_out.state));

                std::string dir_text = "Decision dir: " +
                    std::to_string(static_cast<int>(decision_out.direction));

                std::string urgency_text = "Urgency: " + std::to_string(decision_out.urgency);
                std::string ttc_text = "TTC proxy: " + std::to_string(decision_out.ttc_proxy);
                std::string score_text = "L/R score: " +
                    std::to_string(decision_out.left_score) + " / " +
                    std::to_string(decision_out.right_score);

                cv::putText(debug_view, state_text, cv::Point(20, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);

                cv::putText(debug_view, dir_text, cv::Point(20, 55),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 0), 2);

                cv::putText(debug_view, urgency_text, cv::Point(20, 80),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2);

                cv::putText(debug_view, ttc_text, cv::Point(20, 105),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 0, 255), 2);

                cv::putText(debug_view, score_text, cv::Point(20, 130),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(200, 200, 200), 2);

                udp_debug.sendFrame(debug_view, 75);
            }

            // =========================
            // MPC + low level command
            // =========================
            MpcState state = mpc.computeMpcParameters(target_centerline, birdEyeView);

            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send).count() >= 50)
            {
                last_send = now;

                if (state.is_valid)
                {
                    float steering = mpc.computeSteeringAngle(state, desired_velocity);

                    steering = 1.5f * std::pow(steering, 3) + 10.0f * steering;

                    if (steering <= -25.0f) steering = -25.0f;
                    else if (steering >= 25.0f) steering = 25.0f;

                    int servo = static_cast<int>(std::lround(97.0f + steering));
                    float velocity_cmd = desired_velocity;

                    std::cout << "[LOGIC] steering=" << steering
                              << " servo=" << servo
                              << " distance=" << distance
                              << std::endl;

                    comm.sendCommands(velocity_cmd, servo);
                }
            }
        }
    });

    while (running.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    if (camera_thread.joinable()) camera_thread.join();
    if (mpc_thread.joinable()) mpc_thread.join();

    std::cout << "[LOGIC] Stopped cleanly." << std::endl;
}