#!/usr/bin/env python
"""Calibrate the bad-actor fraction against the collision rate it produces.

An earlier probe showed that pushing 15 % of vehicles to ignore other vehicles
produces 126 collisions per simulated minute -- far more than traffic, and most
of them meaningless contact in stopped queues.  This sweeps the bad-actor
fraction using the behaviour profiles the generator will actually use, and
reports not just how many collisions happen but **how hard they are**, so that a
threshold separating a real crash from a bumper nudge can be chosen from data.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_episode import BEHAVIOURS, BEHAVIOUR_ID, apply_behaviour  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town05")
    ap.add_argument("--vehicles", type=int, default=160)
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--warmup", type=float, default=60.0)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--bad-actor-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import carla
    rng = random.Random(args.seed)
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(180.0)
    world = client.get_world()
    if args.town not in world.get_map().name:
        world = client.load_world(args.town)
    st = world.get_settings()
    st.synchronous_mode = True
    st.fixed_delta_seconds = args.delta
    st.no_rendering_mode = True
    st.max_substep_delta_time = 0.01
    st.max_substeps = 16
    world.apply_settings(st)
    tm = client.get_trafficmanager(args.port + 6000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)

    sp = world.get_map().get_spawn_points()
    rng.shuffle(sp)
    bps = sorted(world.get_blueprint_library().filter("vehicle.*"), key=lambda b: b.id)
    SpawnActor, SetAutopilot = carla.command.SpawnActor, carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    batch = []
    for i in range(min(args.vehicles, len(sp))):
        bp = rng.choice(bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
        batch.append(SpawnActor(bp, sp[i]).then(SetAutopilot(FutureActor, True, args.port + 6000)))
    ids = [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]
    vehicles = list(world.get_actors(ids))

    n_bad = int(round(args.bad_actor_fraction * len(vehicles)))
    order = list(range(len(vehicles)))
    rng.shuffle(order)
    behaviour = ["normal"] * len(vehicles)
    kinds = ["speeding", "red_light_runner", "erratic", "tailgater"]
    for j, idx in enumerate(order[:n_bad]):
        behaviour[idx] = kinds[j % len(kinds)]
    for k, v in enumerate(vehicles):
        apply_behaviour(tm, v, behaviour[k], rng)
    beh_of = {v.id: behaviour[k] for k, v in enumerate(vehicles)}

    events = []
    sensors = []
    cbp = world.get_blueprint_library().find("sensor.other.collision")
    for r, v in zip(client.apply_batch_sync(
            [SpawnActor(cbp, carla.Transform(), v) for v in vehicles], True), vehicles):
        if not r.error:
            s = world.get_actor(r.actor_id)
            sensors.append(s)
            s.listen(lambda e, vid=v.id: events.append(
                (e.frame, vid, e.other_actor.id if e.other_actor else -1,
                 float(e.normal_impulse.length()))))

    for _ in range(int(args.warmup / args.delta)):
        world.tick()
    events.clear()

    # Track speeds so each contact can be labelled with the speed of the faster
    # participant at impact -- the physical difference between a crash and a
    # queue nudge.
    speed_at = {}
    n = int(args.seconds / args.delta)
    for _ in range(n):
        world.tick()
        snap = world.get_snapshot()
        for v in vehicles:
            a = snap.find(v.id)
            if a is not None:
                vel = a.get_velocity()
                speed_at[(snap.frame, v.id)] = (vel.x ** 2 + vel.y ** 2) ** 0.5

    grouped = defaultdict(list)
    for f, a, b, imp in events:
        grouped[(min(a, b), max(a, b))].append((f, a, b, imp))
    distinct = []
    gap = int(1.0 / args.delta)
    for pair, evs in grouped.items():
        evs.sort()
        last = -10 ** 9
        for f, a, b, imp in evs:
            if f - last > gap:
                sa = speed_at.get((f, a), 0.0)
                sb = speed_at.get((f, b), 0.0)
                distinct.append({"impulse": imp, "speed": max(sa, sb),
                                 "behaviour_a": beh_of.get(a, "?"),
                                 "behaviour_b": beh_of.get(b, "?")})
            last = max(last, f)

    imp = np.array([d["impulse"] for d in distinct]) if distinct else np.zeros(0)
    spd = np.array([d["speed"] for d in distinct]) if distinct else np.zeros(0)
    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else 0.0
    minutes = args.seconds / 60.0
    rep = {
        "vehicles": len(vehicles), "bad_actor_fraction": args.bad_actor_fraction,
        "n_bad": n_bad, "sim_minutes": minutes,
        "collisions_total": len(distinct),
        "collisions_per_minute": round(len(distinct) / minutes, 2),
        "impulse": {"p10": pct(imp, 10), "p50": pct(imp, 50), "p90": pct(imp, 90),
                    "max": float(imp.max()) if len(imp) else 0.0},
        "impact_speed_m_s": {"p10": pct(spd, 10), "p50": pct(spd, 50),
                             "p90": pct(spd, 90), "max": float(spd.max()) if len(spd) else 0.0},
    }
    # A "real" crash: someone was actually moving.  The thresholds are reported
    # at several levels so one can be chosen rather than assumed.
    for thr in (1.0, 3.0, 5.0, 8.0):
        rep[f"per_minute_impact_over_{thr:g}ms"] = round(
            float((spd > thr).sum()) / minutes, 2)
    by = defaultdict(int)
    for d in distinct:
        if d["speed"] > 3.0:
            by[tuple(sorted((d["behaviour_a"], d["behaviour_b"])))] += 1
    rep["pairs_over_3ms"] = {" + ".join(k): v for k, v in
                             sorted(by.items(), key=lambda x: -x[1])[:10]}
    print(json.dumps(rep, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(rep, open(args.out, "w"), indent=2)

    for s in sensors:
        try:
            s.stop(); s.destroy()
        except Exception:
            pass
    client.apply_batch([carla.command.DestroyActor(i) for i in ids])
    tm.set_synchronous_mode(False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
