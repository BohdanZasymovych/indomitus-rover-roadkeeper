<h1 align="center">Indomitus — Roadkeeper</h1>

<p align="center">
  Camera-based road keeping for <b>Indomitus</b> &mdash; the Mars rover built by the
  <a href="https://indomitus-ucu.vercel.app/">UCU Space Robotics</a> team at the
  Ukrainian Catholic University for the
  <a href="https://roverchallenge.eu/">European Rover Challenge</a>.
</p>

<p align="center">
  <a href="https://docs.ros.org/en/humble/"><img alt="ROS 2 Humble" src="https://img.shields.io/badge/ROS%202-Humble-22314E?logo=ros&logoColor=white"></a>
  <a href="https://releases.ubuntu.com/22.04/"><img alt="Ubuntu 22.04" src="https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu&logoColor=white"></a>
  <img alt="Platform" src="https://img.shields.io/badge/platform-amd64%20%7C%20arm64-informational">
</p>

<p align="center">
  <a href="https://github.com/BohdanZasymovych/indomitus-rover-roadkeeper/actions/workflows/publish_image.yaml?query=branch%3Adevelop"><img alt="develop build" src="https://img.shields.io/github/actions/workflow/status/BohdanZasymovych/indomitus-rover-roadkeeper/publish_image.yaml?branch=develop&label=develop%20build&logo=github"></a>
  <a href="https://github.com/BohdanZasymovych/indomitus-rover-roadkeeper/releases"><img alt="develop release" src="https://img.shields.io/github/v/release/BohdanZasymovych/indomitus-rover-roadkeeper?include_prereleases&label=develop"></a>
</p>

---


## About

Roadkeeper helps an operator drive the rover along a road by hand. A camera
watches the road ahead, the road is segmented out of each frame, and the
rover's position relative to it is estimated: how far it is from the road
centre and how far its heading is turned away from the road's direction. With
that estimate the rover can warn the operator when it is about to leave the
road, or correct the steering itself while the operator keeps driving.

The operator stays in control the whole time. Roadkeeper only assists manual
driving and does not plan routes or drive on its own.

> **Fork:** This repository is a fork of
> [UCUSpaceRobotics/indomitus-rover-roadkeeper](https://github.com/UCUSpaceRobotics/indomitus-rover-roadkeeper),
> the main Indomitus rover codebase. It keeps the rover's drive, sensors,
> teleop and simulation from there. The robotic arm, the switch panel, Nav2
> autonomous navigation, the LoRa/ground-station comms link and ArUco tracking
> have been removed, because road keeping doesn't need them.


---


## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
- [Demo Videos](#demo-videos)
- [Link Guard](#link-guard)
- [Docker](#docker)
- [Running the Full Pipeline in Simulation](#running-the-full-pipeline-in-simulation)
- [Running the Full Pipeline on the Real Rover](#running-the-full-pipeline-on-the-real-rover)
- [Turning Off the Rover](#turning-off-the-rover)
- [More Documentation](#more-documentation)


---


## Getting Started
1. Read this README
2. Set up [Docker](docker/README.md)
3. Read [CONTRIBUTING](https://github.com/UCUSpaceRobotics/.github/blob/main/.github/CONTRIBUTING.md) before making any changes

The pipeline is:

```
camera ─► road_segmentation_node ─► road_deviation_node ─► road_assist_node ─► twist_mux ─► drive
 image_raw   /rover/perception/       /rover/road/state      cmd_vel_remote_raw   cmd_vel_remote
             road_mask                                       (operator's input)   (corrected)
```

| Package | Role |
| --- | --- |
| `rover_road_segmentation` | DINOv2 road mask from the camera image |
| `rover_road_assist` | road deviation estimate (`road_deviation_node`) and steering correction (`road_assist_node`) |
| `rover_link_guard` | drives the rover to safety when the operator's link is lost |
| `rover_bringup` | `road_pipeline.launch.py` (the chain above) and `rover.launch.py` (the rover itself) |
| `rover_sim` | Gazebo worlds, including the camera-only `dirt_road` simulation |


---


## Demo Videos

| Video | What it shows |
| --- | --- |
| [path_autocorrection.mp4](assets/path_autocorrection.mp4) | The assist correcting the operator's steering to keep the rover on the road |
| [link_loss.mp4](assets/link_loss.mp4) | [Link guard](#link-guard) taking over when the operator's link is lost |
| [real_road_segmentation.mp4](assets/real_road_segmentation.mp4) | Road segmentation on footage of a real road |


---


## Link Guard

`rover_link_guard` (`link_guard_node`) keeps the rover safe when the operator's link drops. It watches `cmd_vel_remote_raw`; if the commands stay silent for more than `lost_after` (1 s), it drives the rover itself on `cmd_vel_ext`, the lowest-priority `twist_mux` input:

1. **Continue:** keep following the road at a reduced speed (using the road estimate), for at most 15 s / 6 m. If the rover was standing still, it stays put.
2. **Retreat:** then back up along the recorded trail to a point before the loss (`retreat: reverse`, or `turn_around` to turn on the spot and drive forward).

The operator takes the rover back just by driving again; the guard stops once commands have arrived for `restore_after` (1 s). Status is on `/rover/link_guard/status`; parameters are in `src/rover/rover_link_guard/config/link_guard.yaml`. It is on by default in `rover.launch.py` and in the simulations; disable it with `link_guard:=false`.


---


## Docker

### Getting the Docker Image

Before doing anything, you need to prepare your local environment:

1. Navigate to the root of the **indomitus-rover-roadkeeper** repository.

2. Copy the example Docker Compose file:

   ```bash
   cp ./docker/docker-compose.dev.example.yaml ./docker-compose.yaml
   ```

> In most cases, the example Docker Compose file should work for you, but you may still need to modify it to match your machine.

Next, you need the Docker image. You can either pull a pre-built image or build it locally.

**Option A: Pull the image from GitHub**

*This may save significant time.*

For example, pull the image from `develop`:

```bash
docker pull ghcr.io/bohdanzasymovych/indomitus-rover-roadkeeper:develop-dev
```

> **Pro Tip:** The images for the `develop` and `main` branches are built automatically, but you can also build an image for any branch. To do so, open the [GitHub Actions page](https://github.com/UCUSpaceRobotics/indomitus-rover-roadkeeper/actions/workflows/publish_image.yaml). Press the gray button on the right, select the branch for which you want to build an image, and click **Run workflow**. When the image is built, you can pull it with the command `docker pull ghcr.io/bohdanzasymovych/indomitus-rover-roadkeeper:<branch-name>-dev` (ensuring any slashes in your branch name are replaced with dashes, like `feature-shared-some-feature-dev`).

**Option B: Build the image locally**

```bash
docker compose build
```


### Start the Container and Build the Workspace

> **⚠️ Important:** All containers mount your local `src/` directory, which means your code will be used inside the container. Ensure that your `src/` directory contains the code you want to test. You may need to switch to the correct branch or pull the code.

1. Start the container in the background:

   ```bash
   docker compose up -d
   ```

   > **⚠️ Important:** By default, this starts the container using the image tag `local-dev` (defined in `docker-compose.yaml`). If you pulled the image from GitHub, it will have a different tag (refer to the [Image Tags](#image-tags) section). To start the container using a different image tag, you need to set the `IMAGE_TAG` environment variable. You can do this either in the `docker-compose.yaml` file or directly in the command:
   > ```bash
   > IMAGE_TAG=feature-shared-some-feature-dev docker compose up -d
   > ```
   > If you omit this, Docker will either build a new image (if there is no image tagged `local-dev`), or it will start the container using the existing `local-dev` image, which may have different dependencies installed.

2. Enter the running container:

   ```bash
   docker exec -it rover_roadkeeper /bin/bash
   ```

3. Build the ROS 2 workspace:

   ```bash
   cd /opt/ws && colcon build --symlink-install && source install/setup.bash
   ```

### Image Tags

Understanding the tag naming convention will help you easily identify the correct Docker image for your hardware and environment.

The `Dockerfile` is split into two primary targets: **`dev`** and **`prod`**. The `prod` target strips out heavy dependencies used for simulations and visualizations (e.g., Gazebo, RViz) to keep the image lightweight for the rover.

Images built locally or by GitHub workflows will use the following tags:

* **`local-prod`**: Production image built locally.
* **`develop-dev`** (or `main-dev`): Development image built continuously by GitHub workflows.
  * *Architecture:* **AMD64**
  * *Use Case:* Can be used on standard Intel/AMD laptops for development and simulation. Cannot run on the Jetson.
* **`develop-prod`** (or `main-prod`): Production image built continuously by GitHub workflows.
  * *Architecture:* **ARM64**
  * *Use Case:* Designed strictly for deployment on the NVIDIA Jetson. Cannot run natively on standard Intel/AMD laptops.
* **`<branch-name>-dev`** and **`<branch-name>-prod`**: Images built by GitHub workflows when you manually trigger **Publish Production and Development Images** for your branch (slashes in the branch name are replaced with dashes).
* **`latest-prod`**: Production image built on merge to the main branch.


---


## Running the Full Pipeline in Simulation

Camera, segmentation, deviation and assist all run against a simulated road in Gazebo. No hardware is needed.

1. Install [Git LFS](src/rover/rover_sim/README.md#git-lfs) and run `git lfs pull` (the simulation meshes are stored in LFS).
2. Get the `dev` image, start the container and enter it (see [Docker](#docker)).
3. Build and source the workspace:
   ```bash
   cd /opt/ws && colcon build --symlink-install && source install/setup.bash
   ```
4. Launch the simulation with the whole pipeline:
   ```bash
   ros2 launch rover_sim road_assist_sim_gz.launch.py
   ```
   This starts Gazebo (`dirt_road` world) with a simulated front camera, the drive stack, the road pipeline and RViz. Useful arguments: `link_guard:=false`, `gui:=false` (headless Gazebo), `rviz:=false`, `assist:=true|false`, `debug_vis:=false`, `device:=cpu|cuda`.
5. In a second terminal (`docker exec -it rover_roadkeeper /bin/bash`), drive the rover:
   ```bash
   ros2 run rover_teleop keyboard_teleop_node   # `r` toggles the assist
   ```
   Or use a gamepad: `ros2 launch rover_teleop gamepad.launch.py`.
6. Inspect the result:
   ```bash
   ros2 topic echo /rover/road/state
   ros2 topic echo /rover/road_assist/status
   ros2 topic echo /rover/link_guard/status
   ```
   To try the [link guard](#link-guard), drive forward and then stop the teleop node: after about a second the rover continues and retreats on its own.
   The road overlay is published on `road_segmentation_node/debug_overlay`.

To test only the steering law, without the camera and segmentation, use the ground-truth variant: `ros2 launch rover_road_assist_sim road_assist_sim.launch.py` ([details](src/rover/rover_road_assist_sim/README.md)).


---


## Running the Full Pipeline on the Real Rover

> **Warning:** Running on the real rover has **not been verified**. The simulation is the only setup in which this pipeline has been tested end to end. The steps below are assembled from the launch files and may need adjustment. Keep the emergency stop within reach and be ready to take over with the joystick.

1. **Power on the rover and connect to its hotspot.** Follow steps 1-2 of the [QUICKSTART](QUICKSTART.md). Wi-Fi: `ERC_UCUSpaceRobotics_A`, password `19283746`.

2. **Deploy the code and image to the Jetson** from your laptop (the `prod` image is built by CI):
   ```bash
   ./scripts/deploy_to_rover.sh pull
   ```
   See [deployment.md](docs/deployment.md) for other modes.

3. **Enter the container on the rover** (a tmux session survives a dropped connection; detach with `Ctrl-b d`):
   ```bash
   ./scripts/enter_container.sh rover core
   ```

4. **Launch the rover with the cameras on.** This starts the drive stack, `road_assist_node` and the camera streams:
   ```bash
   ros2 launch rover_bringup rover.launch.py use_cameras:=true
   ```

5. **Launch the perception chain in a second shell on the rover** (`./scripts/enter_container.sh rover`). `rover.launch.py` already starts the cameras and `road_assist_node`, so skip them here:
   ```bash
   ros2 launch rover_bringup road_pipeline.launch.py use_cameras:=false use_assist:=false \
       image_topic:=/rover/cameras/arducam_front_bottom/image_raw
   ```
   Set `image_topic` to the camera you want to use (the default is `/rover/cameras/zed2i/image_raw`, which needs the ZED camera streaming). Use `device:=cuda` or `device:=cpu` to pick the segmentation device.

6. **Drive from the laptop.** Start the laptop container (see [Docker](#docker); use `ROS_DOMAIN_ID=90` to match the rover), then run:
   ```bash
   ros2 launch rover_teleop gamepad.launch.py
   ```
   The assist starts off by default. Toggle it with the gamepad (`road_assist_button`, see `gamepad.yaml`) or `ros2 service call /rover/road_assist/toggle std_srvs/srv/Trigger`, and watch `/rover/road_assist/status`. The operator can always override it by turning the wheel.

> **Note:** Link guard is on in `rover.launch.py`: if the laptop link drops, the rover continues and then retreats on its own (see [Link Guard](#link-guard)). Test it only with free space around the rover, or start with `link_guard:=false`.

To bring up the pieces without the rest of the rover (e.g. to test the camera and segmentation on a bench), `road_pipeline.launch.py` can also start its own cameras: `ros2 launch rover_bringup road_pipeline.launch.py cameras:=arducam-front-bottom image_topic:=/rover/cameras/arducam_front_bottom/image_raw`.


---


## Turning Off the Rover

> **Warning:** Before cutting power, shut down the Jetson first by SSHing into it and running `sudo poweroff`. Wait about a minute, until the Jetson's fan and LED turn off. Cutting power without doing this can corrupt the Jetson's file system.

To power down the rover, perform **one** of the following actions:

* **Use the power switch:** Rotate the blue switch on the back of the rover counterclockwise.
* **Use the Emergency Stop:** Press one of the two red buttons on the top of the rover. To restart afterwards, release the button by rotating it clockwise.


---


## More Documentation

- [QUICKSTART](QUICKSTART.md) — starting the rover step by step
- [Road segmentation](src/rover/rover_road_segmentation/README.md) and [simulation](src/rover/rover_sim/README.md)
- [Networking / SSH](docs/networking/ssh.md), [deployment](docs/deployment.md), [scripts](docs/scripts/), [software docs](docs/software/), [testing](docs/software/testing.md)
