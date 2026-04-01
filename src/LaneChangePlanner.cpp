#include "LaneChangePlanner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

LaneChangePlanner::LaneChangePlanner()
    : state_(PlannerState::KEEP_LANE),
      last_direction_(DONT_CHANGE),
      hold_counter_(0),// Số frame còn lại để giữ trạng thái đổi lane sau khi đã quyết định đổi lane, tránh việc đổi lane liên tục qua lại
      hold_frames_(10),// Số frame cần giữ trạng thái đổi lane, giá trị này cần tune thử nghiệm để phù hợp với tốc độ và đặc tính của xe
      trigger_distance_(1.1f),// khoảng cách kích hoạt đổi lane, khi obstacle ở khoảng cách này thì planner sẽ bắt đầu cân nhắc đổi lane, giá trị này cần tune thử nghiệm để phù hợp với tốc độ và đặc tính của xe
      lane_width_m_(0.40f),// Giá trị cần tune lại
      vehicle_width_m_(0.21f),// da tune lại cho phù hợp với kích thước của xe, nếu thấy đổi lane sát quá thì tăng thêm, nếu thấy đổi lane quá xa thì giảm bớt đây là chiều rộng của xe mình
      vehicle_length_m_(0.432f),// da tune lại cho phù hợp với kích thước của xe, nếu thấy đổi lane sát quá thì tăng thêm, nếu thấy đổi lane quá xa thì giảm bớt đây là chiều dài của xe mình
      obstacle_width_m_(0.20f),// da tune lại cho phù hợp với kích thước của obstacle, nếu thấy đổi lane sát quá thì tăng thêm, nếu thấy đổi lane quá xa thì giảm bớt đây là chiều rộng ước lượng của obstacle cần tránh, có thể là xe đạp, xe máy hoặc người đi bộ
      obstacle_length_m_(0.22f),// da tune lại cho phù hợp với kích thước của obstacle, nếu thấy đổi lane sát quá thì tăng thêm, nếu thấy đổi lane quá xa thì giảm bớt đây là chiều dài ước lượng của obstacle cần tránh, có thể là xe đạp, xe máy hoặc người đi bộ
      safe_margin_m_(0.08f),// Giá trị cần tune lại nếu thấy xe đổi lane sát quá thì tăng thêm, nếu thấy đổi lane quá xa thì giảm bớt đây là khoảng cách an toàn giữa xe mình với obstacle khi đổi lane
      vx_mps_(0.08f),
      meter_per_pixel_(0.001f),// Giá trị mặc định, sẽ được cập nhật lại khi có lane width hợp lệ
      last_min_distance_m_(std::numeric_limits<float>::infinity()),// khoảng cách nhỏ nhất đến obstacle của quỹ đạo đã chọn ở lần cập nhật trước, dùng để theo dõi và debug
      last_min_ttc_s_(std::numeric_limits<float>::infinity()),// thời gian va chạm nhỏ nhất của quỹ đạo đã chọn ở lần cập nhật trước, dùng để theo dõi và debug
      last_cost_(std::numeric_limits<float>::infinity()),// chi phí của quỹ đạo đã chọn ở lần cập nhật trước, dùng để theo dõi và debug
      last_img_width_(640),
      last_img_height_(480)
{
}
// Các hàm setter để cấu hình các tham số của planner, có kiểm tra giá trị đầu vào để tránh cấu hình sai lệch quá lớn
void LaneChangePlanner::setLaneWidthMeters(float lane_width_m)
{
    if (lane_width_m > 0.05f)
        lane_width_m_ = lane_width_m;
}

void LaneChangePlanner::setVehicleSize(float width_m, float length_m)
{
    if (width_m > 0.05f) vehicle_width_m_ = width_m;
    if (length_m > 0.05f) vehicle_length_m_ = length_m;
}

void LaneChangePlanner::setObstacleSize(float width_m, float length_m)
{
    if (width_m > 0.05f) obstacle_width_m_ = width_m;
    if (length_m > 0.05f) obstacle_length_m_ = length_m;
}

void LaneChangePlanner::setSafeMargin(float safe_margin_m)
{
    if (safe_margin_m >= 0.0f)
        safe_margin_m_ = safe_margin_m;
}

void LaneChangePlanner::setSpeed(float vx_mps)
{
    if (vx_mps > 0.02f)
        vx_mps_ = vx_mps;
}

void LaneChangePlanner::setTriggerDistance(float trigger_distance_m)
{
    if (trigger_distance_m > 0.05f)
        trigger_distance_ = trigger_distance_m;
}

std::vector<cv::Point> LaneChangePlanner::update(
    const std::vector<cv::Point>& base_centerline,
    const cv::Vec3f& /*left_coeff*/,
    const cv::Vec3f& /*right_coeff*/,
    bool has_left_lane,
    bool has_right_lane,
    float lane_width_px,
    LaneLineType left_type,// loại lane line bên trái, dùng để quyết định có được phép đổi lane hay ko
    LaneLineType right_type,// loại lane line bên phải, dùng để quyết định có được phép đổi lane hay ko
    float obstacle_distance,
    int img_width,
    int img_height
)
{
    last_img_width_ = img_width;
    last_img_height_ = img_height;
//Nếu số điểm trên centerline quá ít thì ko đủ để sinh quỹ đạo đổi lane, giữ nguyên lane và trả về centerline gốc
    if (base_centerline.size() < 3)
    {
        state_ = PlannerState::KEEP_LANE;
        last_direction_ = DONT_CHANGE;
        last_min_distance_m_ = std::numeric_limits<float>::infinity();
        last_min_ttc_s_ = std::numeric_limits<float>::infinity();
        last_cost_ = std::numeric_limits<float>::infinity();
        return base_centerline;
    }

    // update data 
    updateScaleFromLaneWidth(lane_width_px);

    std::vector<ReferencePoint> ref = buildReferencePath(base_centerline);
    if (ref.size() < 3)
    {
        state_ = PlannerState::KEEP_LANE;
        last_direction_ = DONT_CHANGE;
        return base_centerline;
    }

    if (hold_counter_ > 0)
        --hold_counter_;

    const bool allow_left = canChangeLeft(obstacle_distance ,has_left_lane, left_type);
    const bool allow_right = canChangeRight(obstacle_distance ,has_right_lane, right_type);

    bool allow_left_committed  = allow_left;
    bool allow_right_committed = allow_right;

    // Nếu đang trong pha CHANGE và đã có hướng trước đó,
    // chỉ cho phép tiếp tục đúng hướng đó
    if (state_ == PlannerState::CHANGE_USING_DASHED)
    {
        if (last_direction_ == CHANGE_LEFT)
        {
            allow_right_committed = false;
        }
        else if (last_direction_ == CHANGE_RIGHT)
        {
            allow_left_committed = false;
        }
    }

    std::vector<Candidate> candidates =
        generateCandidates(ref, allow_left_committed, allow_right_committed);

    // xây dựng 1 đối tượng target cần đánh giá
    StaticObstacle obstacle = buildStaticObstacle(obstacle_distance);
    // xác định cost của từng candidate

    for (auto& c : candidates)
    {
        evaluateCandidate(c, obstacle);
    }

    Candidate best = selectBestCandidate(candidates);

    last_min_distance_m_ = best.min_distance_m;
    last_min_ttc_s_ = best.min_ttc_s;
    last_cost_ = best.cost;

    updatePlannerState(best);

    if (best.polyline_px.size() >= 3)
        return best.polyline_px;

    return base_centerline;
}

void LaneChangePlanner::updateScaleFromLaneWidth(float lane_width_px)
{
    if (lane_width_px > 50.0f && lane_width_m_ > 0.05f)
    {
        meter_per_pixel_ = lane_width_m_ / lane_width_px;
    }
}


// chuyển tập hợp các point -> sang hệ Frenet (s, d)
// khoảng cách dọc, vector pháp tuyến, vị trí pixel
std::vector<LaneChangePlanner::ReferencePoint>
LaneChangePlanner::buildReferencePath(const std::vector<cv::Point>& base_centerline) const
{
    std::vector<ReferencePoint> ref;
    if (base_centerline.size() < 3 || meter_per_pixel_ <= 0.0f)
        return ref;

    std::vector<cv::Point2f> ordered;
    ordered.reserve(base_centerline.size());
    for (auto it = base_centerline.rbegin(); it != base_centerline.rend(); ++it)
    {    // tiêu muốn xét s từ gần đến xa xe
        ordered.emplace_back(static_cast<float>(it->x), static_cast<float>(it->y));
    }

    // xét kích thước
    ref.reserve(ordered.size());

    // biến tích lũy khoảng cách dọc
    float cumulative_s = 0.0f;

    for (size_t i = 0; i < ordered.size(); ++i)
    {
        if (i > 0)
        { // khoảng cách euclid giữa 2 điểm liên tiếp
            cumulative_s += static_cast<float>(cv::norm(ordered[i] - ordered[i - 1])) * meter_per_pixel_;
        }

        cv::Point2f tangent(0.0f, -1.0f);

        // tính vector tiếp tuyến tại từng điểm
        if (i + 1 < ordered.size())
        {
            tangent = ordered[i + 1] - ordered[i];
        }
        else if (i > 0)
        {
            tangent = ordered[i] - ordered[i - 1];
        }

        // chuẩn hóa vector =>> vector tiếp tuyến đơn vị
        float norm_t = std::sqrt(tangent.x * tangent.x + tangent.y * tangent.y);
        if (norm_t < 1e-6f) //  nếu khoảng cách quá nhỏ or nhiễu thì hard code
            tangent = cv::Point2f(0.0f, -1.0f);
        else
            tangent *= (1.0f / norm_t);

        cv::Point2f n_right(-tangent.y, tangent.x);

        ref.push_back({ordered[i], cumulative_s, n_right});
    }

    return ref;
}

LaneChangePlanner::StaticObstacle
LaneChangePlanner::buildStaticObstacle(float obstacle_distance_m) const
{
    StaticObstacle obs;
    obs.s_m = obstacle_distance_m;
    obs.d_m = 0.0f;
    obs.width_m = obstacle_width_m_;
    obs.length_m = obstacle_length_m_;
    obs.valid = (obstacle_distance_m > 0.02f && vx_mps_ > 0.02f);
    return obs;
}
// Ham decision
bool LaneChangePlanner::canChangeLeft(float distance, bool has_left_lane, LaneLineType left_type) const
{
    return (distance <= trigger_distance_) && has_left_lane && left_type == LaneLineType::DASHED;
}
// Ham decision
bool LaneChangePlanner::canChangeRight(float distance,, bool has_right_lane, LaneLineType right_type) const
{
    return(distance <= trigger_distance_) && has_right_lane && right_type == LaneLineType::DASHED;
}

// sinh quỹ đạo
std::vector<LaneChangePlanner::Candidate>
LaneChangePlanner::generateCandidates(
    const std::vector<ReferencePoint>& ref,
    bool allow_left,
    bool allow_right
) const
{
    // tiến hành sinh các quỹ đạo với các case và thời gian khác nhau
    std::vector<Candidate> candidates;

    // giữ lane
    candidates.push_back(makeKeepLaneCandidate(ref));

    // Bộ T phù hợp hơn với vx = 0.08 m/s
    static const std::array<float, 4> T_SET = {0.80f, 1.00f, 1.20f, 1.50f};

    // đổi lane
    if (allow_left)
    {
        for (float T : T_SET)
            candidates.push_back(makeLaneChangeCandidate(ref, -1, T));
    }

    if (allow_right)
    {
        for (float T : T_SET)
            candidates.push_back(makeLaneChangeCandidate(ref, +1, T));
    }

    return candidates;
}

LaneChangePlanner::Candidate
LaneChangePlanner::makeKeepLaneCandidate(const std::vector<ReferencePoint>& ref) const
{
    Candidate c;
    c.target_lane = 0;
    c.maneuver_time_s = 0.0f;
    c.lane_change_distance_m = 0.0f;
    c.target_offset_m = 0.0f;
    c.polyline_px = offsetReferenceToPolyline(ref, c);
    c.feasible = (c.polyline_px.size() >= 3);
    return c;
}

LaneChangePlanner::Candidate
LaneChangePlanner::makeLaneChangeCandidate(
    const std::vector<ReferencePoint>& ref,
    int target_lane,
    float maneuver_time_s
) const
{
    Candidate c;
    c.target_lane = target_lane;
    c.maneuver_time_s = maneuver_time_s;
    c.lane_change_distance_m = std::max(0.15f, vx_mps_ * maneuver_time_s);
    c.target_offset_m = static_cast<float>(target_lane) * lane_width_m_;
    c.polyline_px = offsetReferenceToPolyline(ref, c);
    c.feasible = (c.polyline_px.size() >= 3);
    return c;
}

// đánh giá quỹ đạo
void LaneChangePlanner::evaluateCandidate(
    Candidate& c, // quỹ đạo cần đánh giá
    const StaticObstacle& obstacle // vật cản tĩnh
) const
{
    // đánh giá hợp lệ ko, nếu ko tạo đc polyline or lỗi hình học thì ko cần đánh giá nữa
    if (!c.feasible)
        return;

    // 2 ngưỡng an toàn theo 2 trục ngang và dọc
    const float lat_clear =
        0.5f * (vehicle_width_m_ + obstacle.width_m) + safe_margin_m_;

    const float long_clear =
        0.5f * (vehicle_length_m_ + obstacle.length_m) + safe_margin_m_;

    // bước lấy mẫu thời gian =>> giảm chi phí tính toán
    const float dt = 0.05f;

    // thời gian mô phỏng tổng để đánh giá quỹ đạo,
    // để đảm bảo ktra cả trong lúc đổi làn và cả đoạn sau đó
    const float t_end =
        obstacle.valid
            ? std::max(c.maneuver_time_s + 0.6f, // thời gian chuyển làn
                       obstacle.s_m / std::max(vx_mps_, 0.05f) + 0.8f) // thời gian va chạm
            : (c.maneuver_time_s + 0.8f);

    c.min_distance_m = std::numeric_limits<float>::infinity(); 
    c.min_ttc_s = std::numeric_limits<float>::infinity();
    c.max_curvature = 0.0f;
    c.max_jerk = 0.0f;
    c.collision = false; // chưa va chạm

    for (float t = 0.0f; t <= t_end; t += dt)
    {
        const float s = vx_mps_ * t; // vị trí dọc của xe ego
        const float d = lateralOffsetAtS(s, c); // trả về quỹ đạo ngang của xe dựa trên quintic blend của candidate

        // đạo hàm bậc 1 của offset ngang theo s
        const float ds = lateralDsAtS(s, c); 
        // đạo hàm bậc 2
        const float dss = lateralDssAtS(s, c);
        // tính jerk
        const float jerk = std::fabs(lateralD3dt3AtS(s, c));

        // tính dộ cong =>> nếu độ cong quá lớn thì ko tối ưu
        const float curvature =
            std::fabs(dss) / std::pow(1.0f + ds * ds, 1.5f);

        // cập nhật jerk và curvature
        c.max_curvature = std::max(c.max_curvature, curvature);
        c.max_jerk = std::max(c.max_jerk, jerk);

        // khi hợp lệ mới đánh giá an toàn
        if (obstacle.valid)
        {
            // khoảng cách tương đối giữa obstacle và ego
            const float ds_obs = obstacle.s_m - s;
            const float dd_obs = obstacle.d_m - d; // 0 =>> cùng làn | lớn =>> lệch ngang nhiều

            // khoảng cách euclid giữa tâm ego và obstacle
            const float center_dist = std::sqrt(ds_obs * ds_obs + dd_obs * dd_obs);
            c.min_distance_m = std::min(c.min_distance_m, center_dist); // cập nhật k/cách nhỏ nhất

            // kiểm tra va chạm nếu nhỏ hơn mức an toàn
            if (std::fabs(ds_obs) <= long_clear && std::fabs(dd_obs) <= lat_clear)
            {
                c.collision = true;
            }
            // update ttc nếu ego chưa vượt và có nguy cơ va chạm 
            if (ds_obs > 0.0f && std::fabs(dd_obs) <= lat_clear)
            {
                const float ttc = ds_obs / std::max(vx_mps_, 0.05f);
                c.min_ttc_s = std::min(c.min_ttc_s, ttc);
            }
        }
    }

    if (!obstacle.valid)
    {
        c.min_distance_m = 999.0f;
        c.min_ttc_s = 999.0f;
    }
    else
    {
        if (!std::isfinite(c.min_distance_m))
            c.min_distance_m = 999.0f;

        if (!std::isfinite(c.min_ttc_s))
            c.min_ttc_s = 999.0f;
    }

    //khoảng cách tổi thiểu planner muốn giữ với obstacle
    const float desired_dist = std::max(lat_clear + 0.05f, 0.18f);
    // ngưỡng ttc an toàn
    const float ttc_threshold = 1.5f;

    const float dist_cost =
        (c.min_distance_m >= desired_dist) // so sánh khoảng cách đủ lớn ko
            ? 0.0f
            : sqr((desired_dist - c.min_distance_m) / desired_dist); // nếu có thì bị phạt theo bình phương mức va chạm

    const float ttc_cost =
        (c.min_ttc_s >= ttc_threshold) // so sánh ttc đủ lớn ko
            ? 0.0f
            : sqr((ttc_threshold - c.min_ttc_s) / ttc_threshold); // nếu có cũng sẽ phạt

    const float curvature_cost = c.max_curvature;  // phạt quỹ đạo cong gắt
    const float jerk_cost = c.max_jerk;             // phạt quỹ đạo gây giật
    const float time_cost = c.maneuver_time_s;      // phạt quỹ đạo quá lâu

    float keep_lane_penalty = 0.0f;
    if (obstacle.valid && c.target_lane == 0 && obstacle.s_m < trigger_distance_)
    {   // đánh giá có nguy cơ va chạm mà ko chuyển làn =>> phạt nặng
        keep_lane_penalty += 8.0f * dist_cost;
        keep_lane_penalty += 6.0f * ttc_cost;
        if (c.collision)
            keep_lane_penalty += 500.0f;
    }

    // thuật toán ưu tiên duy trì hướng đổi cũ =>> planner có quán tính quyết định
    float direction_bias = 0.0f;
    if (hold_counter_ > 0)
    {
        if (last_direction_ == CHANGE_LEFT)
        {
            if (c.target_lane == -1)      direction_bias = -7.0f;
            else if (c.target_lane == 0)  direction_bias = +15.0f;  // phạt bỏ đổi lane giữa chừng
            else if (c.target_lane == +1) direction_bias = +7.0f;
        }
        else if (last_direction_ == CHANGE_RIGHT)
        {
            if (c.target_lane == +1)      direction_bias = -7.0f;
            else if (c.target_lane == 0)  direction_bias = +15.0f;
            else if (c.target_lane == -1) direction_bias = +7.0f;
        }
    }

    if (c.collision)
    { // nếu va chạm thif cost cực lớn (hard rejection)
        c.cost = 1.0e6f;
    }
    else
    {
        c.cost =
            12.0f * dist_cost +                 // ko đc quá gần obstancle
            10.0f * ttc_cost +                  // thời gian va chạm
            1.00f * curvature_cost +            // độ cong, 
            0.03f * jerk_cost +                 // độ comfort
            0.25f * time_cost +                 // ưu tiên quỹ đạo ngắn
            keep_lane_penalty +                 // thúc ép đổi khi obs quá gần
            direction_bias;                     // tạo ổn định khi ra quyết định
    }
}

LaneChangePlanner::Candidate
LaneChangePlanner::selectBestCandidate(const std::vector<Candidate>& candidates) const
{
    Candidate best;
    const Candidate* latched = nullptr;

    for (const auto& c : candidates)
    {
        if (!c.feasible)
            continue;

        if (c.cost < best.cost)
            best = c;

        if (hold_counter_ > 0) // hold counter giữ quán tính cho xe
        {
            if (last_direction_ == CHANGE_LEFT && c.target_lane == -1)
                latched = &c;
            else if (last_direction_ == CHANGE_RIGHT && c.target_lane == +1)
                latched = &c;
        }
    }

    if (latched != nullptr && latched->cost <= best.cost + 7.0f)
        return *latched;

    return best;
}

void LaneChangePlanner::updatePlannerState(const Candidate& best)
{
    if (best.target_lane == 0)
    {
        if (state_ == PlannerState::CHANGE_USING_DASHED)
            state_ = PlannerState::FOLLOW_LANE;
        else
            state_ = PlannerState::KEEP_LANE;

        last_direction_ = DONT_CHANGE;
        hold_counter_ = hold_frames_;
        return;
    }

    state_ = PlannerState::CHANGE_USING_DASHED;

    if (best.target_lane < 0)
        last_direction_ = CHANGE_LEFT;
    else
        last_direction_ = CHANGE_RIGHT;

    hold_counter_ = hold_frames_;
}

// 
std::vector<cv::Point> LaneChangePlanner::offsetReferenceToPolyline(
    const std::vector<ReferencePoint>& ref,             // đường tham chiếu xây dựng từ basecenterline
    const Candidate& c                                  // quỹ đạo ứng viên
) const
{
    std::vector<cv::Point> polyline;        // quỹ đạo mới sinh 
    polyline.reserve(ref.size());           // cấp phát bộ nhớ

    for (const auto& rp : ref)
    {
        // độ lệch ngang tại 1 vị trí s
        const float d_m = lateralOffsetAtS(rp.s_m, c);
        // đổi từ hệ met =>> pixel
        const float d_px = d_m / std::max(meter_per_pixel_, 1e-6f);

        // dịch chuyển theo chiều vector pháp tuyến
        float x = rp.pos_px.x + d_px * rp.n_right.x;
        float y = rp.pos_px.y + d_px * rp.n_right.y;

        // làm tròn về dạng số nguyên
        int xi = static_cast<int>(std::lround(x));
        int yi = static_cast<int>(std::lround(y));

        // giới hạn trong biên của ảnh
        xi = std::clamp(xi, 0, std::max(0, last_img_width_ - 1));
        yi = std::clamp(yi, 0, std::max(0, last_img_height_ - 1));

        // thêm điểm vào 
        polyline.emplace_back(xi, yi);
    }
    
    // do trong quá trình reference thì ta đã đảo để xác định s của xe
    std::reverse(polyline.begin(), polyline.end());
    return polyline;
}

float LaneChangePlanner::lateralOffsetAtS(float s_m, const Candidate& c) const
{
    if (c.target_lane == 0 || c.lane_change_distance_m <= 1e-6f)
        return 0.0f;

    if (s_m <= 0.0f)
        return 0.0f;

    if (s_m >= c.lane_change_distance_m)
        return c.target_offset_m;

    const float sigma = s_m / c.lane_change_distance_m;
    return c.target_offset_m * quinticBlend(sigma);
}

float LaneChangePlanner::lateralDsAtS(float s_m, const Candidate& c) const
{
    if (c.target_lane == 0 || c.lane_change_distance_m <= 1e-6f)
        return 0.0f;

    if (s_m <= 0.0f || s_m >= c.lane_change_distance_m)
        return 0.0f;

    const float sigma = s_m / c.lane_change_distance_m;
    return c.target_offset_m * quinticBlendD1(sigma) / c.lane_change_distance_m;
}

float LaneChangePlanner::lateralDssAtS(float s_m, const Candidate& c) const
{
    // ko đổi làn
    if (c.target_lane == 0 || c.lane_change_distance_m <= 1e-6f)
        return 0.0f;

    // đổi xong r or ngoài đoạn đổi làn
    if (s_m <= 0.0f || s_m >= c.lane_change_distance_m)
        return 0.0f;

    // chuẩn hóa sigma = [0,1]
    const float sigma = s_m / c.lane_change_distance_m;
    const float denom = c.lane_change_distance_m * c.lane_change_distance_m;
    return c.target_offset_m * quinticBlendD2(sigma) / denom;
}

float LaneChangePlanner::lateralD3dt3AtS(float s_m, const Candidate& c) const
{
    if (c.target_lane == 0 || c.lane_change_distance_m <= 1e-6f)
        return 0.0f;

    if (s_m <= 0.0f || s_m >= c.lane_change_distance_m)
        return 0.0f;

    const float sigma = s_m / c.lane_change_distance_m;
    const float scale = std::pow(vx_mps_ / c.lane_change_distance_m, 3.0f);
    return c.target_offset_m * quinticBlendD3(sigma) * scale;
}

// sigma = s / total_distance
float LaneChangePlanner::quinticBlend(float sigma)
{
    sigma = std::clamp(sigma, 0.0f, 1.0f);
    return 10.0f * sigma * sigma * sigma
         - 15.0f * sigma * sigma * sigma * sigma
         + 6.0f * sigma * sigma * sigma * sigma * sigma;
}

// vận tốc
float LaneChangePlanner::quinticBlendD1(float sigma)
{
    sigma = std::clamp(sigma, 0.0f, 1.0f);
    return 30.0f * sigma * sigma
         - 60.0f * sigma * sigma * sigma
         + 30.0f * sigma * sigma * sigma * sigma;
}


// gia tốc
float LaneChangePlanner::quinticBlendD2(float sigma)
{
    sigma = std::clamp(sigma, 0.0f, 1.0f);
    return 60.0f * sigma
         - 180.0f * sigma * sigma
         + 120.0f * sigma * sigma * sigma;
}

// jerk
float LaneChangePlanner::quinticBlendD3(float sigma)
{
    sigma = std::clamp(sigma, 0.0f, 1.0f);
    return 60.0f - 360.0f * sigma + 360.0f * sigma * sigma;
}