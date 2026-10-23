# ROVER SIM

This package contains all the assets and configurations required to run the Gazebo simulation.

## Git LFS

> ⚠️ **Important** Do this before working with simulation!

This package includes large mesh assets tracked with [Git LFS](https://git-lfs.github.com/). Install it once per machine before cloning/pulling, otherwise large mesh files will check out as tiny pointer stubs instead of the real content:

**Linux**
```bash
sudo apt update && sudo apt install git-lfs   
git lfs install
```

**Mac OS**
```bash
brew install git-lfs
```

If you already cloned the repo before installing Git LFS, pull the real files with:

```bash
git lfs pull
```

## Worlds & maps

| World | Map years | Resolutions |
|---|---|---|
| `mars_yard` | `2025`, `2026` | `low`, `medium`, `high` (2025 only; ignored for 2026) |
| `nav2_test_world` | - | - |
| `flat_track` | - | - |

## `sim_gz.launch.py`

Launches the Gazebo world, spawns the rover, and starts its base stack: `ros_gz_bridge`, `robot_state_publisher`, EKF, `twist_mux`, `drive_manager`, `road_assist_node`, and the base + swerve controllers.

| Argument | Default | Description |
|---|---|---|
| `swerve_controller` | `swerve_controller_test` | Which swerve controller comes up active (`swerve_controller`, `swerve_controller_test`). |
| `world_name` | `mars_yard` | World to load: `mars_yard`, `nav2_test_world`, `flat_track` (flat 300 m plane, used by `rover_road_assist_sim`). |
| `map_year` | `2026` | `mars_yard` map year: `2025`, `2026`. |
| `map_resolution` | `high` | `mars_yard` (2025) mesh resolution: `low`, `medium`, `high`. Ignored for 2026/other worlds. |
| `model_name` | `indomitus_rover` | Name the robot model is spawned under. |
| `spawn_delay` | `5.0` | Seconds to wait before spawning the robot. |
| `extra_xacro_args` | *(empty)* | Extra flags passed to the URDF xacro compiler. |
| `spawn_x` / `spawn_y` / `spawn_z` | `0.0` / `0.0` / `0.5` | Initial spawn coordinates (m). |
| `road_assist_params_file` | *(empty)* | Params layered over `road_assist.yaml` for the `road_assist_node` this launch starts (it forwards laptop teleop's `cmd_vel_remote_raw` to `cmd_vel_remote`). |

**Example:**

```bash
ros2 launch rover_sim sim_gz.launch.py
```

Legacy/fallback (2025 map):

```bash
ros2 launch rover_sim sim_gz.launch.py map_year:=2025 map_resolution:=high
```

## Camera-only simulation

```bash
ros2 launch rover_sim deftech.launch.py
```

This starts `worlds/deftech.world` with the simulated front RGB camera enabled.
The LiDAR, depth camera, and camera IMU are disabled. The RGB image is bridged
to `/rover/zed2i/rgb/image_rect_color`.