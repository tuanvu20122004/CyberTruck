Files included:
- PolicyModel.hpp / PolicyModel.cpp
- logic_shadow_mode.hpp / logic_shadow_mode.cpp
- main_shadow_mode.cpp

What this does:
- Loads policy_export.json exported by train_raw_steering.py
- Builds the exact same 8 runtime features used during training
- Runs policy inference online to predict raw steering
- Still drives the vehicle using MPC raw steering (shadow mode)
- Logs MPC vs policy predictions to policy_shadow_log.csv

Feature order expected at runtime:
1. lateral_deviation
2. yaw_angle
3. curvature_0
4. curvature_1
5. curvature_2
6. curvature_3
7. velocity
8. prev_steering

Suggested integration steps:
1. Add PolicyModel.hpp/.cpp to your project.
2. Replace your current logic.hpp/.cpp with the shadow-mode versions,
   or merge the PolicyModel and shadow logging parts into your existing Logic class.
3. Place policy_export.json next to the executable, or pass its path as argv[2].
4. Build and run in shadow mode first.

Runtime outputs:
- policy_shadow_log.csv  : structured CSV for analysis
- policy_shadow_log.txt  : text log via existing Logger

When you are satisfied with shadow mode:
- switch actuator control from raw_steering_mpc to raw_steering_pred
- keep MPC running in parallel as a fallback/safety supervisor
