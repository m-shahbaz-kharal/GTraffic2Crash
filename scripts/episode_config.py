"""Deterministic per-episode configuration.

Episode id -> (vehicles, bad-actor fraction, seconds).  Deterministic so that any
episode can be regenerated exactly, and so the mix is a stated design rather than
an accident of scheduling.

Half the episodes are nominal.  Traffic Manager traffic with no bad actors was
measured at about one collision per simulated minute, essentially all of it
low-speed contact in queues, so nominal episodes really are the clean baseline.
The rest carry 6-14 % bad actors, which a calibration probe put at roughly one
meaningful collision per simulated minute.
"""
import numpy as np

RECORD_SECONDS = 450.0
WARMUP_SECONDS = 60.0


def config(episode: int) -> dict:
    rng = np.random.default_rng(20260831 + episode)
    incident = (episode % 2 == 1)
    return {
        "episode": episode,
        "vehicles": int(rng.integers(100, 261)),
        "bad_actor_fraction": float(rng.uniform(0.06, 0.14)) if incident else 0.0,
        "seconds": RECORD_SECONDS,
        "warmup": WARMUP_SECONDS,
        "kind": "incident" if incident else "nominal",
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        n = int(sys.argv[2])
        cs = [config(i) for i in range(n)]
        veh = np.array([c["vehicles"] for c in cs])
        bad = np.array([c["bad_actor_fraction"] for c in cs])
        print(f"{n} episodes, {sum(c['kind']=='incident' for c in cs)} incident")
        print(f"vehicles {veh.min()}-{veh.max()} (mean {veh.mean():.0f})")
        print(f"bad-actor fraction on incident episodes "
              f"{bad[bad>0].min():.3f}-{bad[bad>0].max():.3f}")
        print(f"total recorded traffic {n*RECORD_SECONDS/3600:.1f} h "
              f"({n*RECORD_SECONDS*2:.0f} frames at 2 Hz)")
    else:
        import json
        print(json.dumps(config(int(sys.argv[1]))))
