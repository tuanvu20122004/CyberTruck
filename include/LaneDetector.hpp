#ifndef LANEDETECTOR_HPP
#define LANEDETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

enum class LaneChangeDirection {
    NONE = 0,
    LEFT = 1,
    RIGHT = 2,
    BOTH = 3
};

enum class LaneLineType {
    UNKNOWN = 0,
    SOLID,
    DASHED
};

class LaneDetector {
public:
    LaneDetector(const std::string& videoPath, int width = 640, int height = 480);
    ~LaneDetector();

    cv::Mat mask;

    void FrameResize(const cv::Mat& frame);
    cv::Mat getFrameResize();
    cv::Mat getMask() const;
    cv::Mat getBirdEyeView() const { return bird_eye_view; }

    LaneLineType left_type = LaneLineType::UNKNOWN;
    LaneLineType right_type = LaneLineType::UNKNOWN;

    bool getFrame(cv::Mat& frame_resize);
    bool isOpened() const;
    void processFrame(cv::Mat& frame_resize);

    // ===== Getters for MPC =====
    std::vector<cv::Point> getCenterline() const { return centerline; }
    bool hasValidLane() const { return has_valid_lane_; }

    // ===== Getters for LaneChangePlanner =====
    cv::Vec3f getLeftCoeffs() const { return left_coeffs_; }
    cv::Vec3f getRightCoeffs() const { return right_coeffs_; }
    bool hasLeftLane() const { return has_left_lane_; }
    bool hasRightLane() const { return has_right_lane_; }
    float getLaneWidthPx() const { return lane_width_px_; }

    // ===== Setter for display =====
    void setSteeringInfo(float steering_cmd, int servo_angle) {
        current_steering_cmd_ = steering_cmd;
        current_servo_angle_ = servo_angle;
        has_steering_info_ = true;
    }

    // ===== Setter for MPC display data =====
    void setMpcDisplayData(float curvature, float lateral_dev, float yaw_angle) {
        display_curvature_ = curvature;
        display_lateral_dev_ = lateral_dev;
        display_yaw_angle_ = yaw_angle;
        has_mpc_data_ = true;
    }

private:
    cv::VideoCapture cap;
    cv::Mat frame, frame_resize;
    int width;
    int height;
    bool initialized = false;

    // ===== Display data =====
    float current_steering_cmd_ = 0.0f;
    int current_servo_angle_ = 0;
    bool has_steering_info_ = false;

    float display_curvature_ = 0.0f;
    float display_lateral_dev_ = 0.0f;
    float display_yaw_angle_ = 0.0f;
    bool has_mpc_data_ = false;

    // ===== Lane detection results =====
    std::vector<cv::Point> centerline;
    cv::Mat bird_eye_view;
    bool has_valid_lane_ = false;

    // ===== Extra data for LaneChangePlanner =====
    cv::Vec3f left_coeffs_{0.0f, 0.0f, 0.0f};
    cv::Vec3f right_coeffs_{0.0f, 0.0f, 0.0f};
    bool has_left_lane_ = false;
    bool has_right_lane_ = false;
    float lane_width_px_ = 200.0f;

    // ===== Lane detection methods =====
    cv::Mat applyIPM(cv::Mat& frame);
    cv::Mat processMask(const cv::Mat& bird_eye_view);

    void slidingWindow(const cv::Mat& mask,
                       std::vector<cv::Point>& left_points,
                       std::vector<cv::Point>& right_points,
                       cv::Mat& outImg,
                       int minpix);

    void slidingWindowAdaptive(const cv::Mat& mask,
                               std::vector<cv::Point>& lane_points,
                               cv::Mat& outImg,
                               cv::Vec3f prev_poly);

    cv::Vec3f fitPoly(const std::vector<cv::Point>& points,
                      cv::Mat& outImg,
                      bool isLeft);

    std::vector<cv::Point> computeCenterline(cv::Vec3f coeff_left,
                                             cv::Vec3f coeff_right,
                                             bool has_left,
                                             bool has_right,
                                             cv::Mat& outImg);

    float computeLaneSlope(const cv::Vec3f& coeffs, float y);

    LaneLineType classifyLaneMarking(const cv::Mat& mask,
                                     const cv::Vec3f& coeff,
                                     int band_half_width = 12,
                                     int y_step = 4,
                                     int min_segment_len = 4,
                                     int min_gap_len = 4);

    // void displayInfo(cv::Mat& frame_resize, bool left_ok, bool right_ok);
};

#endif
