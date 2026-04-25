#include <iostream>
#include <stdexcept>
#include <string>
#include "logic_shadow_mode.hpp"

static RunMode parseRunMode(const std::string& s)
{
    if (s == "policy" || s == "policy_rollout") {
        return RunMode::PolicyRollout;
    }
    return RunMode::Shadow;
}

static bool parseBoolFlag(const std::string& s)
{
    return (s == "1" || s == "true" || s == "on" || s == "yes");
}

int main(int argc, char* argv[])
{
    std::cout << "=== Policy rollout with MPC fallback ===\n";

    std::string videoPath  = "/dev/video0";
    std::string policyPath = "bc_train.json";
    std::string modelName  = "bc";
    std::string csvPath    = "rollout_bc.csv";
    std::string modeStr    = "policy_rollout";
    std::string exactQStr  = "true";

    if (argc > 1) videoPath  = argv[1];
    if (argc > 2) policyPath = argv[2];
    if (argc > 3) modelName  = argv[3];
    if (argc > 4) csvPath    = argv[4];
    if (argc > 5) modeStr    = argv[5];
    if (argc > 6) exactQStr  = argv[6];

    try {
        Logic logic(videoPath,
                    policyPath,
                    modelName,
                    csvPath,
                    parseRunMode(modeStr),
                    parseBoolFlag(exactQStr));
        logic.run();
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << '\n';
        return 1;
    }

    return 0;
}