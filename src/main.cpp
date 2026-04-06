#include <iostream>
#include <stdexcept>
#include "logic.hpp"

int main(int argc, char* argv[]) {
    std::cout << "=== Lane Keeping System ===\n";

    std::string videoPath = "/dev/video0";
    ControlMode mode = ControlMode::PURE_PURSUIT; // default

    if (argc > 1) {
        videoPath = argv[1];
    }
    if (argc > 2) {
        std::string mode_str = argv[2];

        if (mode_str == "mpc")
            mode = ControlMode::MPC;
        else if (mode_str == "pp")
            mode = ControlMode::PURE_PURSUIT;
    }

    try {
        Logic logic(videoPath);
        logic.setControlMode(mode);   
        logic.run();
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }

    return 0;
}