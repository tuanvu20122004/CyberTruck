#ifndef MPCCONTROLLER_HPP
#define MPCCONTROLLER_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>
#include <memory>
#include <vector>
#include <limits>

struct MpcState {
    std::vector<float> curvature;
    float lateral_deviation;
    float yaw_angle;
    bool is_valid;

    MpcState()
        : curvature(10, 0.0f),
          lateral_deviation(0.0f),
          yaw_angle(0.0f),
          is_valid(false) {}
};

struct ExactQResult {
    double objective;
    double dQ_du_rad;
    double dQ_du_deg;
    double lambda_u;
    bool valid;

    ExactQResult()
        : objective(std::numeric_limits<double>::infinity()),
          dQ_du_rad(0.0),
          dQ_du_deg(0.0),
          lambda_u(0.0),
          valid(false) {}
};

class MpcController {
public:
    MpcController();

    void init(float Q1_weight, float Q2_weight, float R_weight);
    void debugMatrices();

    MpcState computeMpcParameters(const std::vector<cv::Point>& centerline,
                                  const cv::Mat& birdEyeView);

    float computeSteeringAngle(const MpcState& state, float velocity);

    // Exact-Q interface for offline labeling.
    // u0_fixed_deg is the enforced first steering move in degrees.
    double evaluateExactQ(const MpcState& state, float velocity, double u0_fixed_deg);

    // Exact-Q plus sensitivity dQ/du obtained from the Lagrange multiplier
    // associated with the enforced first action constraint u0 - u0_fixed = 0.
    ExactQResult evaluateExactQWithGradient(const MpcState& state,
                                           float velocity,
                                           double u0_fixed_deg);

    void setVehicleParams(float wheelbase, float mass, float Lf, float Lr,
                          float Caf, float Car, float Iz);
    void setPredictionHorizon(int N);
    void setVehiclePosition(float x, float y);
    void setWeights(float q1, float q2, float r);

private:
    float wheelbase_;
    float mass_;
    float Lf_, Lr_;
    float Caf_, Car_;
    float Iz_;

    int N_;
    float Q1_, Q2_, R_;
    float Ts_;
    bool initialized_;

    Eigen::MatrixXd A_d_, B1_d_, B2_d_;
    Eigen::MatrixXd AX_, BU_, BV_, H_;

    double umin_, umax_;
    std::unique_ptr<OsqpEigen::Solver> solver_;
    bool solver_initialized_;

    float pixel_per_meter_;
    float vehicle_x_;
    float vehicle_y_;

    void buildMpcMatrices(float Vx);
    Eigen::MatrixXd matrixPower(const Eigen::MatrixXd& A, int p);
    float solveQP(const Eigen::VectorXd& x0, const Eigen::VectorXd& v_k);
    double solveQPExactQ(const Eigen::VectorXd& x0,
                         const Eigen::VectorXd& v_k,
                         double u0_fixed_rad);

    ExactQResult solveQPExactQWithGradient(const Eigen::VectorXd& x0,
                                           const Eigen::VectorXd& v_k,
                                           double u0_fixed_rad);

    cv::Vec3f fitCenterlinePoly(const std::vector<cv::Point>& centerline);
    std::vector<float> computeMultipleCurvatures(const cv::Vec3f& coeffs, int N = 10);
    float computeLateralDeviation(const cv::Vec3f& coeffs, const cv::Mat& birdEyeView);
    float computeYawAngle(const cv::Vec3f& coeffs, float y);
};

#endif
