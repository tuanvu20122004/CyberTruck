#ifndef LANECHANGEPLANNER_HPP
#define LANECHANGEPLANNER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "LaneDetector.hpp"

enum class PlannerState {
    KEEP_LANE = 0,
    CHANGE_LEFT,
    FOLLOW_LEFT_LANE,
    CHANGE_RIGHT,
    FOLLOW_RIGHT_LANE
};

class LaneChangePlanner {
public:
    LaneChangePlanner();

    std::vector<cv::Point> update(
        const std::vector<cv::Point>& base_centerline,
        const cv::Vec3f& left_coeff,
        const cv::Vec3f& right_coeff,
        bool has_left_lane,
        bool has_right_lane,
        float lane_width_px,
        LaneLineType left_type,
        LaneLineType right_type,
        float obstacle_distance,
        int img_width,
        int img_height
    );

    PlannerState getState() const { return state_; }

private:
    PlannerState state_;
    float progress_;
    float trigger_distance_;
    float change_rate_;

    std::vector<cv::Point> buildCenterlineFromBoundary(
        const cv::Vec3f& coeff,
        float offset_px,
        int img_width,
        int img_height
    );

    std::vector<cv::Point> blendCenterlines(
        const std::vector<cv::Point>& from_line,
        const std::vector<cv::Point>& to_line,
        float alpha
    );

    float smoothStep(float x);
};

#endif
