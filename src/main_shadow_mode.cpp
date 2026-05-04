#include <iostream>
#include <stdexcept>
#include <string>
#include "logic_shadow_mode.hpp"

int main(int argc, char* argv[])
{
    std::cout << "=== Policy control / DAgger collection with MPC fallback ===\n";

    std::string videoPath = "/dev/video0";
    std::string policyPath = "bc_train.json";
    std::string logPrefix = "dagger_run";
    std::string runMode = "dagger";

    if (argc > 1) {
        videoPath = argv[1];
    }
    if (argc > 2) {
        policyPath = argv[2];
    }
    if (argc > 3) {
        logPrefix = argv[3];
    }
    if (argc > 4) {
        runMode = argv[4];
    }

    std::cout << "videoPath = " << videoPath << '\n'
              << "policyPath = " << policyPath << '\n'
              << "logPrefix = " << logPrefix << '\n'
              << "runMode = " << runMode << '\n';

    try {
        Logic logic(videoPath, policyPath, logPrefix, runMode);
        logic.run();
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << '\n';
        return 1;
    }

    return 0;
}
