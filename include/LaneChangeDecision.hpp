#ifndef LANECHANGEDECISION_HPP
#define LANECHANGEDECISION_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <limits>
#include "LaneDetector.hpp"

enum class DecisionDirection
{
    NONE = 0,
    LEFT,
    RIGHT
};

enum class DecisionState
{
    KEEP_LANE = 0,
    PREPARE_CHANGE,
    APPROVE_CHANGE,
    COOLDOWN
};

class LaneChangeDecision
{
public:
    struct Input
    {
        std::vector<cv::Point> base_centerline;

        cv::Vec3f left_coeff{0, 0, 0};
        cv::Vec3f right_coeff{0, 0, 0};

        bool has_left_lane = false;
        bool has_right_lane = false;

        LaneLineType left_type = LaneLineType::SOLID;
        LaneLineType right_type = LaneLineType::SOLID;

        float lane_width_px = 400.0f;

        // khoảng cách vật cản phía trước, đơn vị mét
        float obstacle_distance = -1.0f;

        // vận tốc xe ego, m/s
        float ego_speed = 0.0f;

        // chu kỳ cập nhật, giây
        float dt = 0.05f;

        int img_width = 640;
        int img_height = 480;
    };

    struct Output
    {
        bool approve_lane_change = false;
        DecisionDirection direction = DecisionDirection::NONE;
        DecisionState state = DecisionState::KEEP_LANE;

        float urgency = 0.0f;          // 0 -> 1
        float ttc_proxy = std::numeric_limits<float>::infinity();
        float left_score = 0.0f;
        float right_score = 0.0f;
        int persistence_count = 0;
    };

    struct Params
    {
        // vùng bắt đầu quan tâm vật cản
        float caution_distance = 2.2f;

        // vùng rất gần -> tăng urgency mạnh
        float critical_distance = 1.1f;

        // TTC proxy nhỏ hơn ngưỡng này thì coi là cấp bách
        float ttc_threshold = 2.2f;

        // số frame liên tiếp đủ điều kiện mới cho đổi làn
        int persistence_frames = 4;

        // số frame khóa sau khi vừa duyệt một lane change
        int cooldown_frames = 18;

        // bias để tránh nhảy trái/phải liên tục
        float hysteresis_bonus = 0.20f;

        // ưu tiên nếu cả hai phía đều hợp lệ
        float keep_direction_bias = 0.12f;
    };

    explicit LaneChangeDecision(const Params& params = Params());

    Output update(const Input& in);

    void notifyLaneChangeStarted();
    void notifyLaneChangeFinished();
    void reset();

private:
    Params params_;

    DecisionState state_;
    DecisionDirection last_preferred_direction_;
    DecisionDirection committed_direction_;

    float last_obstacle_distance_;
    bool has_last_distance_;

    int persistence_count_;
    int cooldown_count_;

    float estimateClosingRate(float distance, float dt);
    float computeTtcProxy(float distance, float closing_rate, float ego_speed) const;
    float computeUrgency(float distance, float ttc_proxy) const;

    std::vector<cv::Point> buildCenterlineFromBoundary(
        const cv::Vec3f& coeff,
        float offset_px,
        int img_width,
        int img_height
    ) const;

    float meanX(const std::vector<cv::Point>& line) const;

    float scoreCandidate(
        const std::vector<cv::Point>& candidate,
        const std::vector<cv::Point>& base_centerline,
        bool lane_visible,
        LaneLineType lane_type,
        float urgency,
        DecisionDirection dir
    ) const;
};

#endif // LANECHANGEDECISION_HPP