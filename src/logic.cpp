#include "logic.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <pthread.h>
#include <cmath>

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
    udp_yolo("192.168.1.100", 9996, 8888),
    udp_debug("192.168.1.100", 9997)
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

            static float last_valid_distance = -1.0f;

            if (udp_yolo.receiveDistance())
            {
                float d = udp_yolo.getDistance();
                if (d > 0.0f)
                    last_valid_distance = d;
            }

            float distance = last_valid_distance;

            int planner_width  = !birdEyeView.empty() ? birdEyeView.cols : frame_local.cols;
            int planner_height = !birdEyeView.empty() ? birdEyeView.rows : frame_local.rows;

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

            auto now_debug = std::chrono::steady_clock::now();
            if (!birdEyeView.empty() &&
                std::chrono::duration_cast<std::chrono::milliseconds>(now_debug - last_debug_send).count() >= 100)
            {
                last_debug_send = now_debug;

                cv::Mat debug_view = birdEyeView.clone();
                Logger::drawPolyline(debug_view, base_centerline, cv::Scalar(255, 0, 0));
                Logger::drawPolyline(debug_view, target_centerline, cv::Scalar(0, 0, 255));

                udp_debug.sendFrame(debug_view, 75);
            }

            MpcState state = mpc.computeMpcParameters(target_centerline, birdEyeView);

            auto now = std::chrono::steady_clock::now();

            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send).count() >= 50)
            {
                last_send = now;

                if (state.is_valid)
                {
                    float steering = mpc.computeSteeringAngle(state, desired_velocity);

                    steering = 0.7f * std::pow(steering, 3) + 1.6f * steering;

                    if (steering <= -25.0f) steering = -25.0f;
                    else if (steering >= 25.0f) steering = 25.0f;

                    int servo = static_cast<int>(std::lround(97.0f + steering));

                    float velocity_cmd = desired_velocity;

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
