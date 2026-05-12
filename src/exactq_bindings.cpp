#include "MpcController.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

class ExactQEvaluator {
public:
    ExactQEvaluator(int horizon,
                    float q1,
                    float q2,
                    float r,
                    float wheelbase,
                    float mass,
                    float lf,
                    float lr,
                    float caf,
                    float car,
                    float iz)
        : horizon_(horizon)
    {
        if (horizon_ <= 0) {
            throw std::runtime_error("horizon must be positive");
        }

        mpc_.setPredictionHorizon(horizon_);
        mpc_.setVehicleParams(wheelbase, mass, lf, lr, caf, car, iz);
        mpc_.init(q1, q2, r);
    }

    py::dict eval_batch(py::array_t<double, py::array::c_style | py::array::forcecast> states,
                        py::array_t<double, py::array::c_style | py::array::forcecast> actions_deg)
    {
        const auto state_buf = states.request();
        const auto action_buf = actions_deg.request();

        if (state_buf.ndim != 2) {
            throw std::runtime_error("states must be a 2-D numpy array");
        }
        if (action_buf.ndim != 1) {
            throw std::runtime_error("actions_deg must be a 1-D numpy array");
        }

        const py::ssize_t batch = state_buf.shape[0];
        const py::ssize_t cols = state_buf.shape[1];
        const py::ssize_t expected_cols = 3 + horizon_;// ey, yaw, curvature[0:N], velocity

        if (cols != expected_cols) {
            throw std::runtime_error("states must have columns [ey, yaw, curvature_0..curvature_Nminus1, velocity]");
        }
        if (action_buf.shape[0] != batch) {
            throw std::runtime_error("actions_deg length must equal states batch size");
        }

        const double* state_ptr = static_cast<const double*>(state_buf.ptr);
        const double* action_ptr = static_cast<const double*>(action_buf.ptr);

        py::array_t<double> q_out(batch);
        py::array_t<double> dq_out(batch);
        py::array_t<double> lambda_out(batch);
        py::array_t<int> valid_out(batch);

        auto q = q_out.mutable_unchecked<1>();
        auto dq = dq_out.mutable_unchecked<1>();
        auto lambda = lambda_out.mutable_unchecked<1>();
        auto valid = valid_out.mutable_unchecked<1>();

        for (py::ssize_t i = 0; i < batch; ++i) {
            const double* row = state_ptr + i * cols;

            MpcState s;
            s.is_valid = true;
            s.lateral_deviation = static_cast<float>(row[0]);
            s.yaw_angle = static_cast<float>(row[1]);
            s.curvature.assign(static_cast<size_t>(horizon_), 0.0f);
            for (int k = 0; k < horizon_; ++k) {
                s.curvature[static_cast<size_t>(k)] = static_cast<float>(row[2 + k]);
            }
            const float velocity = static_cast<float>(row[2 + horizon_]);
            const double u_deg = action_ptr[i];

            const ExactQResult res = mpc_.evaluateExactQWithGradient(s, velocity, u_deg);
            q(i) = res.objective;
            dq(i) = res.dQ_du_deg;
            lambda(i) = res.lambda_u;
            valid(i) = res.valid ? 1 : 0;
        }

        py::dict out;
        out["q"] = q_out;
        out["dq_du_deg"] = dq_out;
        out["lambda_u"] = lambda_out;
        out["valid"] = valid_out;
        return out;
    }

private:
    int horizon_;
    MpcController mpc_;
};

PYBIND11_MODULE(exactq_mpc, m) {
    m.doc() = "Exact-Q evaluator exposing dQ/du from the Lagrange multiplier of u0 = u_fixed";

    py::class_<ExactQEvaluator>(m, "ExactQEvaluator")
        .def(py::init<int, float, float, float, float, float, float, float, float, float, float>(),
             py::arg("horizon") = 10,
             py::arg("q1") = 1000.0f,
             py::arg("q2") = 50.0f,
             py::arg("r") = 5.0f,
             py::arg("wheelbase") = 0.2515f,
             py::arg("mass") = 2.3f,
             py::arg("lf") = 0.132f,
             py::arg("lr") = 0.12f,
             py::arg("caf") = 0.04f,
             py::arg("car") = 0.02f,
             py::arg("iz") = 0.04f)
        .def("eval_batch", &ExactQEvaluator::eval_batch,
             py::arg("states"),
             py::arg("actions_deg"));
}
