#include "MpcController.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string trim(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> out;
    std::string current;
    bool in_quotes = false;

    for (char ch : line) {
        if (ch == '"') {
            in_quotes = !in_quotes;
        } else if (ch == ',' && !in_quotes) {
            out.push_back(trim(current));
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    out.push_back(trim(current));
    return out;
}

bool hasColumn(const std::map<std::string, std::size_t>& idx, const std::string& name) {
    return idx.find(name) != idx.end();
}

double getDouble(const std::vector<std::string>& row,
                 const std::map<std::string, std::size_t>& idx,
                 const std::string& name,
                 double default_value = 0.0) {
    auto it = idx.find(name);
    if (it == idx.end()) {
        return default_value;
    }
    const std::size_t col = it->second;
    if (col >= row.size()) {
        return default_value;
    }
    const std::string val = trim(row[col]);
    if (val.empty()) {
        return default_value;
    }
    try {
        return std::stod(val);
    } catch (...) {
        return default_value;
    }
}

long long getInt64(const std::vector<std::string>& row,
                   const std::map<std::string, std::size_t>& idx,
                   const std::string& name,
                   long long default_value = 0) {
    auto it = idx.find(name);
    if (it == idx.end()) {
        return default_value;
    }
    const std::size_t col = it->second;
    if (col >= row.size()) {
        return default_value;
    }
    const std::string val = trim(row[col]);
    if (val.empty()) {
        return default_value;
    }
    try {
        return std::stoll(val);
    } catch (...) {
        return default_value;
    }
}

int pickFirstAvailable(const std::vector<std::string>& row,
                       const std::map<std::string, std::size_t>& idx,
                       const std::vector<std::string>& names,
                       int fallback = 0) {
    for (const auto& name : names) {
        if (hasColumn(idx, name)) {
            return static_cast<int>(std::lround(getDouble(row, idx, name, fallback)));
        }
    }
    return fallback;
}

double pickFirstAvailableDouble(const std::vector<std::string>& row,
                                const std::map<std::string, std::size_t>& idx,
                                const std::vector<std::string>& names,
                                double fallback = 0.0) {
    for (const auto& name : names) {
        if (hasColumn(idx, name)) {
            return getDouble(row, idx, name, fallback);
        }
    }
    return fallback;
}

MpcState stateFromRow(const std::vector<std::string>& row,
                      const std::map<std::string, std::size_t>& idx,
                      int horizon) {
    MpcState state;
    state.is_valid = true;
    state.lateral_deviation = static_cast<float>(getDouble(row, idx, "lateral_deviation", 0.0));
    state.yaw_angle = static_cast<float>(getDouble(row, idx, "yaw_angle", 0.0));
    state.curvature.assign(horizon, 0.0f);
    for (int i = 0; i < horizon; ++i) {
        const std::string key = "curvature_" + std::to_string(i);
        state.curvature[i] = static_cast<float>(getDouble(row, idx, key, 0.0));
    }
    if (hasColumn(idx, "is_valid")) {
        state.is_valid = (std::lround(getDouble(row, idx, "is_valid", 1.0)) != 0);
    }
    return state;
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <input.csv> <output.csv> [velocity]\n"
              << "  input.csv  : rollout / fallback log\n"
              << "  output.csv : csv with exact-Q columns appended\n"
              << "  velocity   : optional default velocity if CSV lacks the column\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string input_csv = argv[1];
    const std::string output_csv = argv[2];
    const double default_velocity = (argc >= 4) ? std::stod(argv[3]) : 0.08;

    std::ifstream fin(input_csv);
    if (!fin.is_open()) {
        std::cerr << "[ExactQBatch] Cannot open input CSV: " << input_csv << "\n";
        return 1;
    }

    std::ofstream fout(output_csv, std::ios::out | std::ios::trunc);
    if (!fout.is_open()) {
        std::cerr << "[ExactQBatch] Cannot open output CSV: " << output_csv << "\n";
        return 1;
    }

    std::string header_line;
    if (!std::getline(fin, header_line)) {
        std::cerr << "[ExactQBatch] Input CSV is empty\n";
        return 1;
    }

    const std::vector<std::string> header = splitCsvLine(header_line);
    std::map<std::string, std::size_t> idx;
    for (std::size_t i = 0; i < header.size(); ++i) {
        idx[trim(header[i])] = i;
    }

    if (!hasColumn(idx, "lateral_deviation") || !hasColumn(idx, "yaw_angle")) {
        std::cerr << "[ExactQBatch] Missing required state columns\n";
        return 1;
    }

    MpcController mpc;
    mpc.init(1000.0f, 50.0f, 5.0f);
    mpc.setVehicleParams(0.2515f, 2.3f, 0.132f, 0.12f, 0.04f, 0.02f, 0.04f);

    fout << header_line
         << ",q_policy_exact,q_expert_exact,q_gap_exact,exact_q_status\n";
    fout << std::fixed << std::setprecision(10);

    std::string line;
    std::size_t line_no = 1;
    std::size_t n_ok = 0;
    std::size_t n_skip = 0;

    while (std::getline(fin, line)) {
        ++line_no;
        if (trim(line).empty()) {
            continue;
        }

        const std::vector<std::string> row = splitCsvLine(line);
        const MpcState state = stateFromRow(row, idx, 10);
        const double velocity = pickFirstAvailableDouble(row, idx, {"velocity"}, default_velocity);

        const double u_policy = pickFirstAvailableDouble(
            row, idx,
            {"raw_steering_policy", "raw_steering_pred", "policy_steering", "raw_steering"},
            0.0);

        const double u_expert = pickFirstAvailableDouble(
            row, idx,
            {"raw_steering_expert", "raw_steering_mpc", "expert_steering"},
            std::numeric_limits<double>::quiet_NaN());

        double q_policy = std::numeric_limits<double>::infinity();
        double q_expert = std::numeric_limits<double>::infinity();
        double q_gap = std::numeric_limits<double>::infinity();
        std::string status = "ok";

        if (!state.is_valid) {
            status = "invalid_state";
            ++n_skip;
        } else if (!std::isfinite(u_policy) || !std::isfinite(u_expert)) {
            status = "missing_action";
            ++n_skip;
        } else {
            q_policy = mpc.evaluateExactQ(state, static_cast<float>(velocity), u_policy);
            q_expert = mpc.evaluateExactQ(state, static_cast<float>(velocity), u_expert);
            if (!std::isfinite(q_policy) || !std::isfinite(q_expert)) {
                status = "solver_fail";
                ++n_skip;
            } else {
                q_gap = q_policy - q_expert;
                ++n_ok;
            }
        }

        fout << line << ','
             << (std::isfinite(q_policy) ? q_policy : 0.0) << ','
             << (std::isfinite(q_expert) ? q_expert : 0.0) << ','
             << (std::isfinite(q_gap) ? q_gap : 0.0) << ','
             << status << '\n';
    }

    std::cout << "[ExactQBatch] Done. ok=" << n_ok << " skip=" << n_skip
              << " output=" << output_csv << "\n";
    return 0;
}
