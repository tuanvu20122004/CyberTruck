#ifndef LANECHANGEPLANNER_HPP
#define LANECHANGEPLANNER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "LaneDetector.hpp"


enum class PlannerState
{
    KEEP_LANE = 0,
    CHANGE_USING_DASHED,
    FOLLOW_LANE
};

enum Type_Change_t
{
    DONT_CHANGE = 0,
    CHANGE_LEFT,
    CHANGE_RIGHT,
};
typedef struct
{
    Type_Change_t type_change;
    int first_access = 0;
    /* data */
} State_Change_Lane_t;


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
    State_Change_Lane_t State_change_line;
    PlannerState state_;
    float progress_;
    float trigger_distance_;
    // bước tăng progress mỗi lần update
    float min_progress_step_;
    float max_progress_step_;


    std::vector<cv::Point> last_target_line_; // state dashed_lane nhỡ bị mất trong vài frame

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

    static float meanX(const std::vector<cv::Point>& line);

    std::vector<cv::Point> chooseDashedTarget(
        const std::vector<cv::Point>& left_target,
        const std::vector<cv::Point>& right_target,
        const std::vector<cv::Point>& base_centerline
    ) const;

    float computeProgressStep(float obstacle_distance) const;

    float aggressiveBlend(float alpha);

    static float smoothStep(float x);
};

#endif
