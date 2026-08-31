#!/usr/bin/env python
"""Simulate one episode of Town05 traffic and record raw vehicle state.

Records state, not rasters.  Grid resolution, time step, channel layout and
class grouping are all decisions that may need revising; keeping the raw
per-tick vehicle states means any of them can be changed for the cost of a CPU
pass instead of re-running the simulator.

Traffic is driven by CARLA's Traffic Manager.  A configurable fraction of
vehicles are given a "bad actor" profile -- speeding, running red lights, or
erratic lane changes -- and each vehicle carries a persistent behaviour tag so
that training windows can later be indexed and balanced by what happens in them.

Output is one compressed .npz per episode; see `save_episode` for the layout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g2c.grid import CLASS_IDS, CLASS_NAMES  # noqa: E402

# Behaviour profiles.  The tag is stored per vehicle so that a window can be
# labelled by which kinds of driver were present in it.
BEHAVIOURS = ["normal", "speeding", "red_light_runner", "erratic", "tailgater"]
BEHAVIOUR_ID = {b: i for i, b in enumerate(BEHAVIOURS)}


def blueprint_attributes(bp) -> dict:
    """The two attributes categorisation needs, read defensively.

    Blueprint attributes are typed and `as_str()` raises on some of them
    (`has_lights` is a bool), so this reads only what is wanted.
    """
    out = {}
    for key in ("base_type", "number_of_wheels"):
        if bp.has_attribute(key):
            try:
                out[key] = bp.get_attribute(key).as_str()
            except RuntimeError:
                pass
    return out


def categorise(bp_id: str, attributes: dict) -> str:
    """Map a CARLA vehicle blueprint to one of the six dataset classes.

    Follows the upstream UCFTCam2TWorld convention exactly so that the two
    datasets label the same vehicle the same way.
    """
    base = (attributes or {}).get("base_type", "")
    if base in CLASS_IDS:
        return base
    try:
        wheels = int(attributes.get("number_of_wheels", 4))
    except (TypeError, ValueError):
        wheels = 4
    if wheels == 2:
        return "bicycle" if ("bike" in bp_id or "crossbike" in bp_id) else "motorcycle"
    for key in ("bus", "truck", "van"):
        if key in bp_id:
            return key
    return "car"


def apply_behaviour(tm, vehicle, behaviour: str, rng: random.Random):
    """Configure one vehicle's Traffic Manager profile.

    Speed is set through `vehicle_percentage_speed_difference`, where a negative
    value means *faster* than the posted limit.

    The profiles deliberately avoid `ignore_vehicles_percentage` except for the
    impaired driver.  A vehicle that is blind to other vehicles drives straight
    through them and produces a constant stream of meaningless contacts -- an
    early probe measured 126 collisions per simulated minute across 172 of 250
    vehicles, which is carnage rather than traffic.  Leaving collision avoidance
    on and instead breaking the *rules* (running lights, speeding, following too
    close) yields crashes that happen for the reasons real crashes happen, and at
    a rate that can be tuned.
    """
    if behaviour == "normal":
        tm.vehicle_percentage_speed_difference(vehicle, rng.uniform(-5, 20))
        tm.distance_to_leading_vehicle(vehicle, rng.uniform(2.0, 4.5))
        tm.ignore_lights_percentage(vehicle, 0.0)
        tm.ignore_signs_percentage(vehicle, 0.0)
        tm.ignore_vehicles_percentage(vehicle, 0.0)
    elif behaviour == "speeding":
        # Well over the limit with a short following gap: the classic rear-end
        # setup when the vehicle ahead stops for a light.
        tm.vehicle_percentage_speed_difference(vehicle, rng.uniform(-90, -50))
        tm.distance_to_leading_vehicle(vehicle, rng.uniform(0.3, 1.2))
        tm.ignore_signs_percentage(vehicle, 90.0)
        tm.ignore_lights_percentage(vehicle, 0.0)
        tm.ignore_vehicles_percentage(vehicle, 0.0)
    elif behaviour == "red_light_runner":
        # Enters junctions against the signal.  Collision avoidance stays on, so
        # a crash only happens when cross traffic is genuinely unavoidable --
        # which is precisely the junction conflict worth learning.
        tm.ignore_lights_percentage(vehicle, 100.0)
        tm.ignore_signs_percentage(vehicle, 100.0)
        tm.vehicle_percentage_speed_difference(vehicle, rng.uniform(-50, -20))
        tm.distance_to_leading_vehicle(vehicle, rng.uniform(1.0, 2.5))
        tm.ignore_vehicles_percentage(vehicle, 0.0)
    elif behaviour == "erratic":
        # The impaired driver: weaves between lanes, judges gaps badly, and is
        # genuinely slow to notice other vehicles.  This is the one profile where
        # partial blindness to other traffic is the point rather than an artefact.
        tm.random_left_lanechange_percentage(vehicle, int(rng.uniform(40, 90)))
        tm.random_right_lanechange_percentage(vehicle, int(rng.uniform(40, 90)))
        tm.ignore_vehicles_percentage(vehicle, rng.uniform(10, 35))
        tm.ignore_lights_percentage(vehicle, rng.uniform(5, 25))
        tm.vehicle_percentage_speed_difference(vehicle, rng.uniform(-50, 30))
        tm.distance_to_leading_vehicle(vehicle, rng.uniform(0.2, 1.5))
    elif behaviour == "tailgater":
        tm.distance_to_leading_vehicle(vehicle, 0.0)
        tm.vehicle_percentage_speed_difference(vehicle, rng.uniform(-35, -10))
        tm.ignore_lights_percentage(vehicle, 0.0)
        tm.ignore_vehicles_percentage(vehicle, 0.0)


def save_episode(path, meta, veh, frames, lights, collisions):
    """Write one episode.

    Vehicle attributes that never change (class, footprint, behaviour) live in a
    per-episode table; the per-tick records hold only what moves.  That keeps a
    7-minute episode at a few tens of megabytes instead of a few hundred.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        meta=json.dumps(meta),
        vehicle_actor_id=np.asarray(veh["actor_id"], np.int32),
        vehicle_class=np.asarray(veh["class"], np.int8),
        vehicle_extent=np.asarray(veh["extent"], np.float32),
        vehicle_is_static=np.asarray(veh["is_static"], np.bool_),
        vehicle_behaviour=np.asarray(veh["behaviour"], np.int8),
        vehicle_blueprint=json.dumps(veh["blueprint"]),
        frame_time=np.asarray(frames["time"], np.float32),
        frame_offset=np.asarray(frames["offset"], np.int64),
        record_vehicle=np.concatenate(frames["vehicle"]).astype(np.int32)
        if frames["vehicle"] else np.zeros(0, np.int32),
        record_xy=np.concatenate(frames["xy"]).astype(np.float32)
        if frames["xy"] else np.zeros((0, 2), np.float32),
        record_yaw=np.concatenate(frames["yaw"]).astype(np.float32)
        if frames["yaw"] else np.zeros(0, np.float32),
        record_velocity=np.concatenate(frames["vel"]).astype(np.float32)
        if frames["vel"] else np.zeros((0, 2), np.float32),
        light_xy=np.asarray(lights["xy"], np.float32),
        light_state=np.asarray(lights["state"], np.uint8),
        collision_frame=np.asarray(collisions["frame"], np.int32),
        collision_a=np.asarray(collisions["a"], np.int32),
        collision_b=np.asarray(collisions["b"], np.int32),
        collision_impulse=np.asarray(collisions["impulse"], np.float32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=0, help="0 = rpc port + 6000")
    ap.add_argument("--town", default="Town05")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--seed", type=int, default=-1, help="-1 = derive from episode id")
    ap.add_argument("--vehicles", type=int, default=140)
    ap.add_argument("--seconds", type=float, default=450.0, help="recorded simulated seconds")
    ap.add_argument("--warmup", type=float, default=90.0)
    ap.add_argument("--delta", type=float, default=0.05, help="simulator timestep")
    ap.add_argument("--record-every", type=int, default=2, help="ticks per stored frame")
    ap.add_argument("--bad-actor-fraction", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    import carla

    seed = args.seed if args.seed >= 0 else 10_000 + args.episode
    rng = random.Random(seed)
    t_start = time.time()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    if args.town not in world.get_map().name:
        world = client.load_world(args.town)

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.delta
    settings.no_rendering_mode = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 16
    world.apply_settings(settings)

    tm_port = args.tm_port or (args.port + 6000)
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)

    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)
    blueprints = sorted(world.get_blueprint_library().filter("vehicle.*"),
                        key=lambda b: b.id)

    SpawnActor, SetAutopilot = carla.command.SpawnActor, carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    batch, chosen = [], []
    for i in range(min(args.vehicles, len(spawn_points))):
        bp = rng.choice(blueprints)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
        bp.set_attribute("role_name", "autopilot")
        chosen.append(bp)
        batch.append(SpawnActor(bp, spawn_points[i]).then(SetAutopilot(FutureActor, True, tm_port)))

    actor_ids, bps = [], []
    for res, bp in zip(client.apply_batch_sync(batch, True), chosen):
        if not res.error:
            actor_ids.append(res.actor_id)
            bps.append(bp)
    vehicles = list(world.get_actors(actor_ids))
    by_id = {v.id: k for k, v in enumerate(vehicles)}
    print(f"[gen] episode {args.episode}: spawned {len(vehicles)} vehicles "
          f"(seed {seed})", flush=True)

    # Behaviour assignment.  Bad actors are drawn without replacement so the
    # requested fraction is exact rather than expected.
    n_bad = int(round(args.bad_actor_fraction * len(vehicles)))
    order = list(range(len(vehicles)))
    rng.shuffle(order)
    behaviour = ["normal"] * len(vehicles)
    bad_kinds = ["speeding", "red_light_runner", "erratic", "tailgater"]
    for j, idx in enumerate(order[:n_bad]):
        behaviour[idx] = bad_kinds[j % len(bad_kinds)]
    for k, v in enumerate(vehicles):
        apply_behaviour(tm, v, behaviour[k], rng)
    counts = {b: behaviour.count(b) for b in BEHAVIOURS}
    print(f"[gen] behaviours: {counts}", flush=True)

    veh = {
        "actor_id": [v.id for v in vehicles],
        "class": [CLASS_NAMES.index(categorise(bp.id, blueprint_attributes(bp)))
                  for bp in bps],
        "extent": [[2 * v.bounding_box.extent.x, 2 * v.bounding_box.extent.y] for v in vehicles],
        "is_static": [False] * len(vehicles),
        "behaviour": [BEHAVIOUR_ID[b] for b in behaviour],
        "blueprint": [bp.id for bp in bps],
    }

    # Collision sensors.  A single contact fires several events and fires on both
    # participants, so raw events are de-duplicated after the run.
    raw_collisions = []
    sensors = []
    cbp = world.get_blueprint_library().find("sensor.other.collision")
    cbatch = [SpawnActor(cbp, carla.Transform(), v) for v in vehicles]
    for res, v in zip(client.apply_batch_sync(cbatch, True), vehicles):
        if not res.error:
            s = world.get_actor(res.actor_id)
            sensors.append(s)
            s.listen(lambda e, vid=v.id: raw_collisions.append(
                (e.frame, vid, e.other_actor.id if e.other_actor else -1,
                 float(e.normal_impulse.length()))))
    print(f"[gen] {len(sensors)} collision sensors attached", flush=True)

    traffic_lights = list(world.get_actors().filter("traffic.traffic_light"))
    lights = {"xy": [[tl.get_location().x, tl.get_location().y] for tl in traffic_lights],
              "state": []}

    for _ in range(int(args.warmup / args.delta)):
        world.tick()
    raw_collisions.clear()
    print(f"[gen] warm-up done ({args.warmup:.0f} s)", flush=True)

    frames = {"time": [], "offset": [0], "vehicle": [], "xy": [], "yaw": [], "vel": []}
    n_ticks = int(args.seconds / args.delta)
    first_frame = None
    frame_of_sim_frame = {}
    total = 0
    for k in range(n_ticks):
        world.tick()
        if k % args.record_every:
            continue
        snap = world.get_snapshot()
        if first_frame is None:
            first_frame = snap.frame
        frame_of_sim_frame[snap.frame] = len(frames["time"])
        idx, xy, yaw, vel = [], [], [], []
        for j, v in enumerate(vehicles):
            a = snap.find(v.id)
            if a is None:
                continue
            tr = a.get_transform()
            ve = a.get_velocity()
            idx.append(j)
            xy.append((tr.location.x, tr.location.y))
            yaw.append(tr.rotation.yaw)
            vel.append((ve.x, ve.y))
        frames["time"].append(snap.timestamp.elapsed_seconds)
        frames["vehicle"].append(np.asarray(idx, np.int32))
        frames["xy"].append(np.asarray(xy, np.float32).reshape(-1, 2))
        frames["yaw"].append(np.asarray(yaw, np.float32))
        frames["vel"].append(np.asarray(vel, np.float32).reshape(-1, 2))
        total += len(idx)
        frames["offset"].append(total)
        lights["state"].append([int(tl.get_state()) for tl in traffic_lights])

    # De-duplicate: keep one record per (pair, contact), where contacts more than
    # one second apart count as separate events.
    grouped = defaultdict(list)
    for sim_frame, a, b, imp in raw_collisions:
        grouped[(min(a, b), max(a, b))].append((sim_frame, a, b, imp))
    collisions = {"frame": [], "a": [], "b": [], "impulse": []}
    gap = int(1.0 / args.delta)
    for pair, evs in grouped.items():
        evs.sort()
        last = -10 ** 9
        for sim_frame, a, b, imp in evs:
            if sim_frame - last > gap:
                # Attribute to the nearest recorded frame at or before the impact.
                cand = [f for f in frame_of_sim_frame if f <= sim_frame]
                fi = frame_of_sim_frame[max(cand)] if cand else 0
                collisions["frame"].append(fi)
                collisions["a"].append(by_id.get(a, -1))
                collisions["b"].append(by_id.get(b, -1))
                collisions["impulse"].append(imp)
            last = max(last, sim_frame)

    meta = {
        "episode": args.episode, "seed": seed, "town": args.town,
        "simulator_dt_s": args.delta, "record_every_ticks": args.record_every,
        "frame_dt_s": args.delta * args.record_every,
        "recorded_seconds": args.seconds, "warmup_seconds": args.warmup,
        "n_vehicles": len(vehicles), "n_frames": len(frames["time"]),
        "bad_actor_fraction": args.bad_actor_fraction,
        "behaviour_counts": counts, "behaviours": BEHAVIOURS,
        "classes": CLASS_NAMES, "n_traffic_lights": len(traffic_lights),
        "n_collisions": len(collisions["frame"]),
        "collision_events_raw": len(raw_collisions),
        "wall_seconds": round(time.time() - t_start, 1),
        "carla_version": client.get_server_version(),
    }
    save_episode(args.out, meta, veh, frames, lights, collisions)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[gen] episode {args.episode}: {meta['n_frames']} frames, "
          f"{meta['n_collisions']} distinct collisions "
          f"({meta['collision_events_raw']} raw events), "
          f"{size_mb:.1f} MB, {meta['wall_seconds']:.0f} s wall", flush=True)

    for s in sensors:
        try:
            s.stop(); s.destroy()
        except Exception:
            pass
    client.apply_batch([carla.command.DestroyActor(i) for i in actor_ids])
    world.apply_settings(original)
    tm.set_synchronous_mode(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
