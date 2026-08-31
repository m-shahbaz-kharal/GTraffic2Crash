#!/usr/bin/env python
"""Measure what CARLA Town05 can actually deliver, before committing to a data plan.

Three questions decide the shape of the dataset, and none of them can be answered
from the documentation:

  throughput   how many simulated seconds per wall-clock second can one server
               produce at a realistic vehicle count, with and without rendering?
               This sets how many hours of traffic are affordable.
  crashes      do Traffic-Manager-driven vehicles actually collide when told to
               ignore lights and other vehicles, and at what rate?  If they do
               not, the bad-actor design has to change.
  readout      how expensive is pulling every vehicle's state out per tick?

Writes a JSON summary so several configurations can be compared.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town05")
    ap.add_argument("--vehicles", type=int, default=200)
    ap.add_argument("--seconds", type=float, default=60.0, help="simulated seconds to run")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--aggressive-frac", type=float, default=0.0)
    ap.add_argument("--collision-sensors", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="simulated seconds to settle before measuring; vehicles "
                         "spawn at rest, so a short warm-up reports a jam that is "
                         "really just the fleet accelerating")
    ap.add_argument("--out", default="")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    import carla

    rep = {"args": vars(args)}
    t0 = time.time()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    rep["server_version"] = client.get_server_version()

    world = client.get_world()
    if args.town not in world.get_map().name:
        print(f"[probe] loading {args.town} ...", flush=True)
        world = client.load_world(args.town)
    rep["map"] = world.get_map().name
    rep["load_seconds"] = round(time.time() - t0, 2)

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.delta
    settings.no_rendering_mode = bool(args.no_render)
    # Physics substepping must satisfy delta <= max_substep_delta_time * max_substeps.
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 16
    world.apply_settings(settings)

    tm = client.get_trafficmanager(args.port + 6000 if args.port < 60000 else 8000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    tm_port = tm.get_port()
    rep["tm_port"] = tm_port

    spawn_points = world.get_map().get_spawn_points()
    rep["n_spawn_points"] = len(spawn_points)
    print(f"[probe] map {rep['map']}, {len(spawn_points)} spawn points, "
          f"no_rendering={args.no_render}", flush=True)

    random.seed(args.seed)
    bps = [b for b in world.get_blueprint_library().filter("vehicle.*")]
    random.shuffle(spawn_points)

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    batch = []
    n_try = min(args.vehicles, len(spawn_points))
    for i in range(n_try):
        bp = random.choice(bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        bp.set_attribute("role_name", "autopilot")
        batch.append(SpawnActor(bp, spawn_points[i]).then(SetAutopilot(FutureActor, True, tm_port)))

    t1 = time.time()
    ids = []
    for r in client.apply_batch_sync(batch, True):
        if not r.error:
            ids.append(r.actor_id)
    rep["spawn_seconds"] = round(time.time() - t1, 2)
    rep["n_spawned"] = len(ids)
    print(f"[probe] spawned {len(ids)}/{n_try} vehicles in {rep['spawn_seconds']}s", flush=True)

    vehicles = world.get_actors(ids)
    n_bad = int(round(args.aggressive_frac * len(ids)))
    bad_ids = set(ids[:n_bad])
    for v in vehicles:
        if v.id in bad_ids:
            tm.ignore_lights_percentage(v, 100.0)
            tm.ignore_signs_percentage(v, 100.0)
            tm.ignore_vehicles_percentage(v, 80.0)
            tm.vehicle_percentage_speed_difference(v, -70.0)   # negative -> faster
            tm.distance_to_leading_vehicle(v, 0.0)
            tm.random_left_lanechange_percentage(v, 60)
            tm.random_right_lanechange_percentage(v, 60)
        else:
            tm.vehicle_percentage_speed_difference(v, random.uniform(-10, 20))
            tm.distance_to_leading_vehicle(v, random.uniform(1.5, 3.5))
    rep["n_aggressive"] = n_bad
    print(f"[probe] {n_bad} aggressive vehicles configured", flush=True)

    collisions = []
    sensors = []
    if args.collision_sensors:
        cbp = world.get_blueprint_library().find("sensor.other.collision")
        t2 = time.time()
        cbatch = [SpawnActor(cbp, carla.Transform(), v) for v in vehicles]
        for r, v in zip(client.apply_batch_sync(cbatch, True), vehicles):
            if not r.error:
                s = world.get_actor(r.actor_id)
                sensors.append(s)
                s.listen(lambda e, vid=v.id: collisions.append(
                    (e.frame, vid, e.other_actor.id if e.other_actor else -1,
                     float(e.normal_impulse.length()))))
        rep["collision_sensor_spawn_seconds"] = round(time.time() - t2, 2)
        print(f"[probe] {len(sensors)} collision sensors attached in "
              f"{rep['collision_sensor_spawn_seconds']}s", flush=True)

    # Warm-up: vehicles spawn at rest, so measuring immediately reports an
    # artefact.  Trace the speed distribution through the warm-up so the right
    # settling time can be read off rather than guessed.
    n_warm = int(args.warmup / args.delta)
    trace = []
    for k in range(n_warm):
        world.tick()
        if k % max(1, int(5.0 / args.delta)) == 0:
            snap = world.get_snapshot()
            sp = []
            for v in vehicles:
                a = snap.find(v.id)
                if a is not None:
                    vel = a.get_velocity()
                    sp.append((vel.x ** 2 + vel.y ** 2) ** 0.5)
            sp.sort()
            trace.append({"sim_time_s": round(k * args.delta, 1),
                          "mean": round(sum(sp) / max(len(sp), 1), 2),
                          "median": round(sp[len(sp) // 2], 2) if sp else 0,
                          "moving_frac": round(sum(x > 0.5 for x in sp) / max(len(sp), 1), 3)})
    rep["warmup_trace"] = trace
    collisions.clear()

    n_ticks = int(args.seconds / args.delta)
    read_time = 0.0
    speeds = []
    t3 = time.time()
    for k in range(n_ticks):
        world.tick()
        t4 = time.time()
        snap = world.get_snapshot()
        rows = 0
        for v in vehicles:
            a = snap.find(v.id)
            if a is None:
                continue
            tr = a.get_transform()
            vel = a.get_velocity()
            rows += 1
            if k % 100 == 0:
                speeds.append((vel.x ** 2 + vel.y ** 2) ** 0.5)
        read_time += time.time() - t4
    wall = time.time() - t3

    rep["ticks"] = n_ticks
    rep["wall_seconds"] = round(wall, 2)
    rep["ticks_per_wall_second"] = round(n_ticks / wall, 2)
    rep["sim_seconds_per_wall_second"] = round(args.seconds / wall, 3)
    rep["state_readout_seconds"] = round(read_time, 2)
    rep["state_readout_fraction"] = round(read_time / wall, 3)
    rep["vehicles_read_per_tick"] = rows
    if speeds:
        speeds.sort()
        rep["moving_fraction"] = round(sum(x > 0.5 for x in speeds) / len(speeds), 3)
        rep["fraction_over_5ms"] = round(sum(x > 5.0 for x in speeds) / len(speeds), 3)
        rep["speed_m_s"] = {"mean": round(sum(speeds) / len(speeds), 2),
                            "p50": round(speeds[len(speeds) // 2], 2),
                            "p95": round(speeds[int(0.95 * len(speeds))], 2),
                            "max": round(speeds[-1], 2)}

    # Collapse repeated sensor events from the same contact into distinct events.
    events = defaultdict(list)
    for frame, vid, other, imp in collisions:
        events[(min(vid, other), max(vid, other))].append(frame)
    distinct = 0
    for k_, frames in events.items():
        frames.sort()
        last = -10 ** 9
        for f in frames:
            if f - last > int(1.0 / args.delta):   # same contact if within 1 s
                distinct += 1
            last = f
    rep["collision_events_raw"] = len(collisions)
    rep["collision_events_distinct"] = distinct
    rep["collisions_per_sim_minute"] = round(distinct / (args.seconds / 60.0), 2)
    rep["vehicle_pairs_involved"] = len(events)

    print(json.dumps(rep, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)

    for s in sensors:
        try:
            s.stop(); s.destroy()
        except Exception:
            pass
    client.apply_batch([carla.command.DestroyActor(i) for i in ids])
    world.apply_settings(original)
    tm.set_synchronous_mode(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
