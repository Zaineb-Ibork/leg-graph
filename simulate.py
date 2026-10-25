"""Generate walking data with NeuroMechFly (FlyGym 2.x) under CPG control.

For each gait (tripod, tetrapod, wave) and several random seeds, the fly walks on
flat ground. Recorded at `--record-hz`:
  - measured joint angles of the 42 actuated leg DoFs (7 per leg),
  - ground contact flag and normal force of each leg,
  - thorax position.

Usage:
    python simulate.py --out data --duration 2.0 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import trange

from flygym import Simulation
from flygym.anatomy import BodySegment
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    CPGController,
    CPGNetwork,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    get_cpg_biases,
    make_locomotion_fly,
)

GAITS = ("tripod", "tetrapod", "wave")


def make_cpg(gait: str, timestep: float, seed: int, coupling: float) -> CPGNetwork:
    biases = get_cpg_biases(gait)
    return CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=np.ones(6) * 12.0,
        intrinsic_amps=np.ones(6),
        coupling_weights=(biases > 0) * coupling,
        phase_biases=biases,
        convergence_coefs=np.ones(6) * 20.0,
        seed=seed,
    )


def run_episode(gait: str, seed: int, duration: float, record_hz: float) -> dict:
    fly = make_locomotion_fly(name="fly", add_adhesion=True)
    world = FlatGroundWorld()
    world.add_fly(fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))
    sim = Simulation(world)

    steps = PreprogrammedSteps()
    dof_order = fly.get_actuated_jointdofs_order("position")
    all_dofs = fly.get_jointdofs_order()
    idx = [all_dofs.index(d) for d in dof_order]  # actuated DoFs among all DoFs
    legs = fly.get_legs_order()
    # group actuated DoFs by leg (7 per leg) in the leg order of contact sensors
    leg_of = [d.child.name[:2] for d in dof_order]

    rng = np.random.default_rng(seed)
    coupling = float(rng.uniform(6.0, 14.0))  # mild variability across episodes
    controller = CPGController(make_cpg(gait, sim.timestep, seed, coupling), steps, dof_order)

    sim.reset()
    apply_locomotion_action(
        sim, fly.name,
        LocomotionAction(steps.default_pose_by_dof_order(dof_order), np.ones(6, bool)),
    )
    sim.warmup()

    n = int(duration / sim.timestep)
    every = max(1, int(round(1.0 / (record_hz * sim.timestep))))
    thorax = fly.get_bodysegs_order().index(BodySegment("c_thorax"))
    rec = {"angles": [], "contact": [], "force": [], "thorax": [], "phase": []}
    for i in trange(n, desc=f"{gait} seed={seed}", leave=False):
        apply_locomotion_action(sim, fly.name, controller.step())
        sim.step()
        if i % every == 0:
            found, forces, *_ = sim.get_ground_contact_info(fly.name)
            rec["angles"].append(sim.get_joint_angles(fly.name)[idx].copy())
            rec["contact"].append(found.copy())
            rec["force"].append(np.linalg.norm(forces, axis=1))
            rec["thorax"].append(sim.get_body_positions(fly.name)[thorax].copy())
            rec["phase"].append(controller.cpg_network.curr_phases % (2 * np.pi))
    sim.close()
    out = {k: np.asarray(v, dtype=np.float32) for k, v in rec.items()}
    out.update(
        dt=np.float32(every * sim.timestep),
        coupling=np.float32(coupling),
        dof_names=np.array([str(d) for d in dof_order]),
        leg_names=np.array(legs),
        dof_leg=np.array(leg_of),
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--duration", type=float, default=2.0)
    p.add_argument("--record-hz", type=float, default=500.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--gaits", nargs="+", default=list(GAITS))
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for gait in a.gaits:
        for seed in a.seeds:
            ep = run_episode(gait, seed, a.duration, a.record_hz)
            np.savez_compressed(Path(a.out) / f"{gait}_seed{seed}.npz", gait=gait, **ep)
            disp = ep["thorax"][-1, 0] - ep["thorax"][0, 0]
            print(f"{gait:9s} seed={seed}  samples={len(ep['angles'])}  forward={disp:.2f} mm")


if __name__ == "__main__":
    main()
