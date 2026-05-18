#include <iostream>
#include <stdexcept>
#include <string>
#include "logic_shadow_mode.hpp"

static void printUsage(const char* app)
{
    std::cout
        << "Usage:\n"
        << "  " << app << " <videoPath> <policyJson> <logPrefix> <runMode>\n\n"
        << "runMode:\n"
        << "  shadow : MPC controls, policy only logged\n"
        << "  dagger : policy controls when safe, MPC fallback\n"
        << "  policy : policy controls full-time when state is valid\n"
        << "  expert : MPC controls, expert dataset logging\n\n"
        << "Examples:\n"
        << "  " << app << " /dev/video0 bc_10curv.json rollout_bc_shadow shadow\n"
        << "  " << app << " /dev/video0 bc_10curv.json rollout_bc_dagger dagger\n"
        << "  " << app << " /dev/video0 bc_10curv.json rollout_bc_policy policy\n"
        << "  " << app << " /dev/video0 exactq_10curv.json rollout_exactq_shadow shadow\n"
        << "  " << app << " /dev/video0 exactq_10curv.json rollout_exactq_dagger dagger\n"
        << "  " << app << " /dev/video0 exactq_10curv.json rollout_exactq_policy policy\n";
}

int main(int argc, char* argv[])
{
    std::cout << "=== Policy Rollout / Shadow / DAgger / Full Policy ===\n";

    if (argc == 2) {
        std::string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    std::string videoPath = "/dev/video0";
    std::string policyPath = "exactq_train_opencv.json";
    std::string logPrefix = "rollout";
    std::string runMode = "policy";

    if (argc > 1) videoPath = argv[1];
    if (argc > 2) policyPath = argv[2];
    if (argc > 3) logPrefix = argv[3];
    if (argc > 4) runMode = argv[4];

    if (runMode != "shadow" &&
        runMode != "dagger" &&
        runMode != "policy" &&
        runMode != "expert") {
        std::cerr << "[FATAL] Invalid runMode: " << runMode << "\n";
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "videoPath  = " << videoPath << '\n'
              << "policyPath = " << policyPath << '\n'
              << "logPrefix  = " << logPrefix << '\n'
              << "runMode    = " << runMode << '\n';

    try {
        Logic logic(videoPath, policyPath, logPrefix, runMode);
        logic.run();
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << '\n';
        return 1;
    }

    return 0;
}