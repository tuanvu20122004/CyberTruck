#include <iostream>
#include <stdexcept>
#include "logic.hpp"

int main(int argc, char* argv[]) {
    std::cout << "=== Lane Keeping System ===\n";

    std::string videoPath = "/dev/video0";   // cố định
    ControlMode mode = ControlMode::MPC; // default
    if (argc > 1) {
        std::string mode_str = argv[1];
        if (mode_str == "mpc")
            mode = ControlMode::MPC;
        else if (mode_str == "pp")
            mode = ControlMode::PURE_PURSUIT;
        else {
            std::cerr << "[ERROR] Invalid mode! Use: mpc or pp\n";
            return 1;
        }
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