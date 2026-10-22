# rover_road_assist_sim

Gazebo test bench for `rover_road_assist` without the camera pipeline: the rover drives a
generated road, and `road_sim_node` computes `road/state` from Gazebo's ground-truth pose.
Simulation only — nothing here is part of the rover bringup.

```bash
ros2 launch rover_road_assist_sim road_assist_sim.launch.py
ros2 run rover_teleop keyboard_teleop_node        # another terminal; r toggles the assist
```

```
teleop -> cmd_vel_remote_raw -> road_assist_node -> cmd_vel_remote -> twist_mux -> drive
                                      ^
Gazebo true pose -> road_sim_node -> road/state
```

## What the launch starts

- `rover_sim/sim_gz.launch.py` on the `flat_track` world, with `simulate_ground_truth:=true`
  (the rover publishes its true pose on the Gazebo topic `/model/<model_name>/odometry`)
- the road, generated from the config and spawned as a visual-only model
- a bridge for that one topic -> `ground_truth/odom`
- `road_sim_node`: ground truth + track -> `road/state`, plus RViz markers
- `road_assist_node` is started by `sim_gz` itself (laptops cannot drive any sim without
  it); this launch passes it the config's `road_assist_node` section, with
  `nominal_road_width` defaulting to the simulated `road_width`, through `sim_gz`'s
  `road_assist_params_file`
- RViz (`rviz:=false` to skip)

## Config — `config/road_sim.yaml`

| group | what |
|---|---|
| topics | `ground_truth_topic`, `road_state_topic`, `markers_topic`, `frame_id` |
| road | `road_width` (also the assist's `nominal_road_width` unless set), `lookahead`, view limits |
| `track` | `start`, `scale`, `main` and one `branch` as `"straight L"` / `"left R deg"` / `"right R deg"` |
| `noise` | `enabled`, std of offset / heading / confidence, `seed` |
| `latency` | `enabled`, `delay` — the estimate is stamped with the pose time, so it arrives this old |
| `confidence` | `mode: fixed` (always `fixed_value`) or `model` (factors below) |

Confidence model — a product of factors in [0, 1], each switchable:

- **lateral**: falls from 0.7 to 1.3 half-widths off centre (road edge leaving the image)
- **heading**: falls from 0.35 to 0.7 rad off the road direction (road leaving the view)
- **curvature**: falls for curves tighter than R 10 m, to 0 at R 4 m, within `curvature_preview` ahead
- **fork**: × `fork_confidence` while both roads of the fork are under the rover

One-run overrides: `noise:=true`, `latency:=true`, `confidence:=model`, `config:=<other.yaml>`.

## Notes

- The road is drawn in RViz in `odom`. In simulation that starts at the spawn point, but it is
  the EKF's estimate, so the robot model can drift slightly off the drawn road; the blue trail
  is the true path, measured exactly like `road/state`.
- At a fork the "camera" follows whichever road the rover is on; steer to choose.
- `max_road_age` in `road_assist.yaml` is 0.3 s: a `latency.delay` near it makes estimates STALE.
