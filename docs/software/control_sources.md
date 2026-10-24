# Control sources

- **Source** = driver + its own interpreter. It sends only a `Twist` on its own `cmd_vel_<source>`
  and `SetBool` / `.../toggle` requests to state owners. Sources know nothing about each other.
- **twist_mux alone picks who drives** (`rover_drive/config/twist_mux.yaml`). Keep
  `drive_source_lamp_node`'s `DEFAULT_SOURCES` in sync by hand.
- **Owners hold state** (`drive_manager`, lights, cameras) and publish it latched; sources read it
  back, never keep a copy.
- **Rule A:** each source has its own raw topic and node names (two pads on one `joy` → a toggle
  fires twice and undoes itself).
- **Rule B:** one publisher per twist_mux input.

| source | raw input | twist_mux | priority |
|---|---|---|---|
| rover gamepad (wired) | `/rover/joy` | `cmd_vel_joy` | 100 |
| ground station | `/gs/joy`, `/gs/switches` | `cmd_vel_gs` | 50 |
| laptop gamepad **or** keyboard (one at a time) | `/rover/remote/joy`, `/rover/remote/keyboard/joy` | `cmd_vel_remote`, via `road_assist_node`¹ | 30 |
| link guard (autonomous fallback) | `cmd_vel_remote_raw` silent, `odom`, `road/state` | `cmd_vel_ext` | 10 |

¹ Laptop teleop publishes `cmd_vel_remote_raw`; `road_assist_node` (rover_road_assist, started by
  `rover.launch.py` and `sim_gz.launch.py`) is the only publisher on `cmd_vel_remote` and passes it
  through, correcting `angular.z` only while the assist is on. Topic names: `road_assist.yaml`.

## Link loss

`link_guard_node` (rover_link_guard, started by `rover.launch.py`, `sim_gz.launch.py` and `road_assist_sim_gz.launch.py`; `link_guard:=false` leaves it out) treats
silence on `cmd_vel_remote_raw` as a lost link (the laptop teleop publishes at a steady rate, even
with the sticks centred). While it lasts it is the only publisher on `cmd_vel_ext`; being the
lowest priority, any operator source takes the rover back just by driving again.

1. **Continue** along the road at a crawl (steering by the same law as the assist), because the
   dead zone often ends ahead. Only if the rover was moving when the link dropped.
2. **Retreat** along the recorded trail to a point `retreat_distance` before where the link was
   lost: `retreat: reverse` backs up it, `retreat: turn_around` turns on the spot first.
3. **Hold** (zero command) until the link returns.

Continue ends at `continue_time` / `continue_distance`, or when the road is not trusted for
`road_lost_grace`. The whole episode ends in a hold after `max_duration`. Status (latched):
`link_guard/status` = `LINK_OK`, `CONTINUE`, `TURN`, `RETREAT`, `HOLD`. Limits: `rover_link_guard/config/link_guard.yaml`.

- Laptop nodes stay in `/rover` with a `remote_` prefix, so relative service names hit the rover.
- Start: rover pad — `rover.launch.py` (`source:=rover`); laptop pad —
  `ros2 launch rover_teleop gamepad.launch.py` (default `source:=remote`); keyboard —
  `ros2 run rover_teleop keyboard_teleop_node`.
- New source: own topics/names → `twist_mux.yaml` + `DEFAULT_SOURCES` → requests only → a row above.
- Open: GS can set the traffic lamps directly, but `drive_source_lamp_node` overwrites red/green
  on the next change.
