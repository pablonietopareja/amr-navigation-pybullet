# Autonomous Mobile Robot Navigation in PyBullet

Autonomous mobile robot simulation combining **A\* path planning**, **particle-filter localization**, **simulated LiDAR**, waypoint tracking and dynamic obstacle handling in a PyBullet environment.

The project demonstrates an end-to-end mobile robotics pipeline in which the robot plans multi-goal missions, estimates its own pose, follows planned trajectories and reacts to moving workers without using ground-truth pose for navigation decisions.

---

## Project Overview

The navigation system combines:

- **A\*** global path planning on occupancy-grid maps
- **Particle Filter localization**
- Differential-drive **odometry**
- Simulated **LiDAR** measurements
- Waypoint-based trajectory tracking
- Multi-goal mission execution
- Dynamic moving obstacles representing workers
- Safety-stop behaviour around workers
- Navigation logging and quantitative performance evaluation

The robot navigates using the **Particle Filter pose estimate** rather than the simulator's ground-truth pose.

Ground truth is used only for simulation, sensor generation and performance evaluation.

---

## Navigation Example

![Multi-goal navigation](pictures/V5/multipleTask_path_v5.png)

The robot plans and executes a sequence of navigation targets while continuously estimating its pose and monitoring the environment for dynamic obstacles.

---

## System Architecture

```text
                Occupancy Grid Map
                       │
                       ▼
                 A* Path Planner
                       │
                       ▼
                  Waypoint Path
                       │
                       ▼
              Navigation Controller
                       │
                       ▼
                    Robot
                  /        \
                 /          \
          Odometry      Simulated LiDAR
                 \          /
                  \        /
                       ▼
                Particle Filter
                       │
                       ▼
              Estimated Robot Pose
                       │
                       └──────────────► Navigation Controller


 Dynamic Workers
       │
       ▼
  LiDAR Detection
       │
       ▼
 Safety Stop / Resume
