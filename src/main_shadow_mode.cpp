#include <iostream>
#include <stdexcept>
#include "logic_shadow_mode.hpp"


int main(int argc, char* argv[])
{
    std::cout << "=== Shadow mode: policy prediction + MPC control ===\n";

    std::string videoPath = "/dev/video0";
    std::string policyPath = "policy_export.json";

    if (argc > 1) {
        videoPath = argv[1];
    }
    if (argc > 2) {
        policyPath = argv[2];
    }

    try {
        Logic logic(videoPath, policyPath);
        logic.run();
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << '\n';
        return 1;
    }

    return 0;
}
