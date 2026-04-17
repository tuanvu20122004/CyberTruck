#include "rollout_schema.hpp"
#include <cmath>
#include <sstream>

RolloutCsvLogger::RolloutCsvLogger(const RolloutMetadata& meta, const std::string& filename)
    : meta_(meta)
{
    file_.open(filename);
    writeHeader();
}

RolloutCsvLogger::~RolloutCsvLogger() {
    if (file_.is_open()) file_.close();
}

std::string RolloutCsvLogger::escape(const std::string& s) {
    if (s.find(',') == std::string::npos) return s;
    return "\"" + s + "\"";
}

void RolloutCsvLogger::writeHeader() {
    file_ <<
    "run_id,method_name,model_tag,scenario_id,lap_id,"
    "timestamp_ms,time_s,frame_id,control_tick_id,"
    "is_valid,lane_width_px,left_lane_detected,right_lane_detected,"
    "lateral_deviation,yaw_angle,curvature_0,curvature_1,curvature_2,curvature_3,velocity,prev_steering_raw,"
    "raw_steering_policy,raw_steering_expert,raw_steering_cmd,raw_abs_error,"
    "delta_policy,delta_expert,delta_cmd,delta_mismatch,"
    "ey_abs,yaw_abs,kappa_max,"
    "steering_sent,servo_command,speed_cmd,speed_measured,"
    "fallback_to_mpc,fallback_reason,monitor_warning,monitor_reason,"
    "stop_event_flag,human_intervention_flag,run_abort_flag,"
    "lateral_error_ref,yaw_error_ref,"
    "x_vehicle,y_vehicle,yaw_vehicle,x_ref,y_ref,yaw_ref,"
    "segment_type,"
    "exact_q_policy,exact_q_expert,q_gap\n";
}

void RolloutCsvLogger::logRow(const RolloutRow& r) {
    file_ <<
    r.run_id << "," << r.method_name << "," << r.model_tag << "," << r.scenario_id << "," << r.lap_id << ","
    << r.timestamp_ms << "," << r.time_s << "," << r.frame_id << "," << r.control_tick_id << ","
    << r.is_valid << "," << r.lane_width_px << "," << r.left_lane_detected << "," << r.right_lane_detected << ","
    << r.lateral_deviation << "," << r.yaw_angle << ","
    << r.curvature[0] << "," << r.curvature[1] << "," << r.curvature[2] << "," << r.curvature[3] << ","
    << r.velocity << "," << r.prev_steering_raw << ","
    << r.raw_steering_policy << "," << r.raw_steering_expert << "," << r.raw_steering_cmd << "," << r.raw_abs_error << ","
    << r.delta_policy << "," << r.delta_expert << "," << r.delta_cmd << "," << r.delta_mismatch << ","
    << r.ey_abs << "," << r.yaw_abs << "," << r.kappa_max << ","
    << r.steering_sent << "," << r.servo_command << "," << r.speed_cmd << "," << r.speed_measured << ","
    << r.fallback_to_mpc << "," << escape(r.fallback_reason) << ","
    << r.monitor_warning << "," << escape(r.monitor_reason) << ","
    << r.stop_event_flag << "," << r.human_intervention_flag << "," << r.run_abort_flag << ","
    << r.lateral_error_ref << "," << r.yaw_error_ref << ","
    << r.x_vehicle << "," << r.y_vehicle << "," << r.yaw_vehicle << ","
    << r.x_ref << "," << r.y_ref << "," << r.yaw_ref << ","
    << escape(r.segment_type) << ","
    << r.exact_q_policy << "," << r.exact_q_expert << "," << r.q_gap
    << "\n";
}