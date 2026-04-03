#ifndef LANECHANGEPLANNER_HPP
#define LANECHANGEPLANNER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <limits>
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

class LaneChangePlanner
{
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
    Type_Change_t getLastDirection() const { return last_direction_; }

    float getLastMinDistance() const { return last_min_distance_m_; }
    float getLastMinTTC() const { return last_min_ttc_s_; }
    float getLastCost() const { return last_cost_; }
    float getMeterPerPixel() const { return meter_per_pixel_; }

    void setLaneWidthMeters(float lane_width_m);
    void setVehicleSize(float width_m, float length_m);
    void setObstacleSize(float width_m, float length_m);
    void setSafeMargin(float safe_margin_m);
    void setSpeed(float vx_mps);
    void setTriggerDistance(float trigger_distance_m);

private:
    struct ReferencePoint
    {
        cv::Point2f pos_px;     // điểm gốc trên centerline
        float s_m;              // khoảng cách dọc 
        cv::Point2f n_right;    // vertor pháp tuyến
    };

    struct StaticObstacle
    {
        float s_m;                  // vị trí dọc của obstacle trong hệ frenet
        float d_m;                  // vị trí ngang----------------------------
        float width_m;              // kích thước 
        float length_m;             // kích thước
        bool valid;                 // obctacle hợp lệ không
    };

    struct Candidate
    {
        int target_lane;                    // -1: left, 0: keep, +1: right
        float maneuver_time_s;              // thời gian đổi lane
        float lane_change_distance_m;       // quãng đường dọc cần để hoàn tất đổi lane
        float target_offset_m;              // độ lệch ngang mục tiêu

        bool feasible;                      // quỹ đạo có hợp lệ hình học ko
        bool collision;                     // đánh giá va chạm chưa?      
        int settle_counter_;
        int settle_frames_;
        // các tiêu chí đánh giá quỹ đạo
        float min_distance_m;
        float min_ttc_s;
        float max_curvature;
        float max_jerk;
        float cost;

        std::vector<cv::Point> polyline_px;

        Candidate()
            : target_lane(0),
              maneuver_time_s(0.0f),
              lane_change_distance_m(0.0f),
              target_offset_m(0.0f),
              feasible(false),
              collision(false),
              min_distance_m(std::numeric_limits<float>::infinity()),
              min_ttc_s(std::numeric_limits<float>::infinity()),
              max_curvature(0.0f),
              max_jerk(0.0f),
              cost(std::numeric_limits<float>::infinity()) {}
    };

private:
    PlannerState state_;
    Type_Change_t last_direction_;

    int hold_counter_;
    int hold_frames_;

    float trigger_distance_;

    float lane_width_m_;
    float vehicle_width_m_;
    float vehicle_length_m_;
    float obstacle_width_m_;
    float obstacle_length_m_;
    float safe_margin_m_;
    float vx_mps_;
    int settle_counter_;
    int settle_frames_;
    float meter_per_pixel_;

    float last_min_distance_m_;
    float last_min_ttc_s_;
    float last_cost_;

    int last_img_width_;
    int last_img_height_;

private:
    void updateScaleFromLaneWidth(float lane_width_px);

    std::vector<ReferencePoint> buildReferencePath(
        const std::vector<cv::Point>& base_centerline
    ) const;

    StaticObstacle buildStaticObstacle(float obstacle_distance_m) const;

    bool canChangeLeft(float distance, bool has_left_lane, LaneLineType left_type) const;
    bool canChangeRight(float distance, bool has_right_lane, LaneLineType right_type) const;

    std::vector<Candidate> generateCandidates(
        const std::vector<ReferencePoint>& ref,
        bool allow_left,
        bool allow_right
    ) const;

    Candidate makeKeepLaneCandidate(
        const std::vector<ReferencePoint>& ref
    ) const;

    Candidate makeLaneChangeCandidate(
        const std::vector<ReferencePoint>& ref,
        int target_lane,
        float maneuver_time_s
    ) const;

    void evaluateCandidate(
        Candidate& candidate,
        const StaticObstacle& obstacle
    ) const;

    Candidate selectBestCandidate(
        const std::vector<Candidate>& candidates
    ) const;

    void updatePlannerState(const Candidate& best);

    std::vector<cv::Point> offsetReferenceToPolyline(
        const std::vector<ReferencePoint>& ref,
        const Candidate& c
    ) const;

    float lateralOffsetAtS(float s_m, const Candidate& c) const;
    float lateralDsAtS(float s_m, const Candidate& c) const;
    float lateralDssAtS(float s_m, const Candidate& c) const;
    float lateralD3dt3AtS(float s_m, const Candidate& c) const;

    static float quinticBlend(float sigma);
    static float quinticBlendD1(float sigma);
    static float quinticBlendD2(float sigma);
    static float quinticBlendD3(float sigma);

    static float sqr(float x) { return x * x; }
};

#endif