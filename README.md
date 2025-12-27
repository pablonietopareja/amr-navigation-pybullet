# FINAL_PROJECT_V5 — Mobile Robot Navigation (A* + Particle Filter + Dynamic Workers) #

# ----------------------------------------------- #

Author: Pablo Nieto Pareja
Course: Automation / Mobile Robotics (Case Study / Final Project)
Version: 5
Last updated: 2025-12-26

# ----------------------------------------------- #

# ---------------- #
# PROJECT PURPOSE: #
# ---------------- #
This project implements an autonomous mobile robot simulation in PyBullet. The robot navigates a 2D grid map using A* path planning and follows the planned route using a Particle Filter (PF) for localization (odometry + simulated lidar). The system supports multi-goal missions and includes dynamic moving obstacles (“workers”) that force the robot to stop and wait when they are too close, improving safety behavior.

# ----------------------------------------------- #

# ---------- #
# HOW TO RUN #
# ---------- #
1) Run the simulation (main)
From the project root folder:
    -> python main.py

This will:
• Load the selected map from maps/
• Spawn the robot in PyBullet
• Run localization (odometry + particle filter)
• Plan paths to multiple targets using A*
• Execute the mission and generate logs in logs/

2) Generate plots (after running main)
Trajectory + planned paths + map overlay:
    -> python plot_path.py

Navigation metrics plots (distance-to-path, distance-to-goal, timing summaries, etc.):
    -> python plot_nav_metrics.py

# ----------------------------------------------- #

# ------------ #
# Output files #
# ------------ #
After running main.py, the following CSV logs are generated in logs/:
• slam_log.csv → time series of true pose, odometry pose, PF pose, current segment/target
• planned_paths.csv → planned A* waypoints for each segment
• nav_metrics.csv → distance-to-path and distance-to-goal metrics over time
• targets_summary.csv → per-target summary (planning time, travel time, path length, etc.)

Plots are saved/displayed by the plotting scripts (depending on your plotting code configuration).

# ----------------------------------------------- #

# ---------------------------- #
# MAIN COMPONENTS (HIGH-LEVEL) #
# ---------------------------- #
• Planning: A* grid planner generates waypoint paths per target segment.
• Localization: Differential-drive odometry prediction + PF correction using simulated lidar scans.
• Control: Pure-pursuit style waypoint tracking with a heading controller.
• Dynamic obstacles: Two moving “worker” spheres detected by ray-based lidar logic; the robot stops when too close and continues once clear.

# ----------------------------------------------- #

# ---------------- #
# FOLDER STRUCTURE #
# ---------------- #
FINAL_PROJECT_V5/
├─ logs/
│  ├─ nav_metrics.csv
│  ├─ planned_paths.csv
│  ├─ slam_log.csv
│  └─ targets_summary.csv
├─ maps/
│  ├─ demoMap
│  ├─ myMap
│  └─ testMap
├─ models/
│  └─ plane.urdf
├─ pictures/
│  ├─ OLD_onlyTask1_corner.png
│  ├─ V1/
│  ├─ V2/
│  ├─ V3/
│  ├─ V4/
│  └─ V5/
│     ├─ logs/
│     │  ├─ nav_metrics.csv
│     │  ├─ planned_paths.csv
│     │  ├─ slam_log.csv
│     │  └─ targets_summary.csv
│     ├─ distanceGoal_v5.png
│     ├─ lateralDeviation_v5.png
│     ├─ multipleTask_path_v5.png
│     ├─ pathDistance_v5.png
│     ├─ robotTime_v5.png
│     ├─ timeAstar_v5.png
│     └─ results_v5.txt
├─ textures/
│  ├─ box.png
│  ├─ red.png
│  ├─ sily.png
│  └─ wall.png
├─ utils/
│  ├─ controller.py
│  ├─ demo_odometry.py
│  ├─ demo_robot.py
│  ├─ mission_runner.py
│  ├─ nav_utils.py
│  ├─ particle_filter.py
│  ├─ planner_astar.py
│  ├─ ScannableMap.py
│  ├─ workers.py
│  └─ world.py
├─ main.py
├─ plot_nav_metrics.py
├─ plot_path.py
├─ timings_0.json
├─ timings_1.json
├─ timings_2.json
├─ timings_3.json
├─ requirements_conda.yml
├─ demo_show.mp4
└─ report.pdf

# ----------------------------------------------- #

# ----- #
# NOTES #
# ----- #
The robot navigation uses the Particle Filter estimate for control and waypoint tracking (no ground-truth pose is used for navigation decisions). Ground-truth pose is only used in simulation to generate lidar measurements and for evaluation/plots.

Map obstacles are defined by ASCII maps in maps/ (e.g., 1 and 2 represent walls/boxes with different textures).

requirements_conda.yml includes everything needed for the Anaconda enviroment to run the project properly, execute this in Anaconda prompt to install it:
    -> conda env create -f requirements_conda.yml
    -> conda activate py311

If you want to install it with a different name then:
    -> conda env create -f requirements_conda.yml -n new_name
    -> conda activate new_name