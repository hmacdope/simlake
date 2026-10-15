"""Make large synthetic trajectories for benchmarking, from numpy noise.

Atoms start uniformly in a cubic box at roughly water density and take a Gaussian
random walk (``--step`` Angstrom per frame). The topology is minimal but real enough
for groups and selections: the first ``--protein`` fraction of atoms are 10-atom ALA
residues (N CA C O CB H HA HB1 HB2 HB3, so ``protein`` and ``backbone`` work), the rest
3-atom SOL waters (OW HW1 HW2).

Read speed on these files is representative. Compression is a worst case: there is no
local structure for XTC (or anything else) to exploit. Use a real trajectory such as
YiiP for compression numbers.

Run, for example:

    python benchmarks/make_trajectories.py --atoms 100000 --frames 5000
    python benchmarks/make_trajectories.py --atoms 1000000 --frames 500

Writes ``<name>.xtc`` and ``<name>.gro`` (topology and first frame) under ``--out``.
"""

import argparse
import os
import time
import warnings

import numpy as np
import MDAnalysis as mda

warnings.filterwarnings("ignore")

ALA = ["N", "CA", "C", "O", "CB", "H", "HA", "HB1", "HB2", "HB3"]
SOL = ["OW", "HW1", "HW2"]


def noise_universe(n_atoms, protein_fraction=0.1, density=0.1, seed=0):
    """A Universe with an ALA "protein" and SOL "water", atoms placed uniformly in a box."""
    n_protein = int(n_atoms * protein_fraction) // len(ALA) * len(ALA)
    n_water = (n_atoms - n_protein) // len(SOL) * len(SOL)
    n_atoms = n_protein + n_water
    n_ala, n_sol = n_protein // len(ALA), n_water // len(SOL)

    resindex = np.concatenate([np.repeat(np.arange(n_ala), len(ALA)),
                               n_ala + np.repeat(np.arange(n_sol), len(SOL))])
    u = mda.Universe.empty(n_atoms, n_residues=n_ala + n_sol, atom_resindex=resindex, trajectory=True)
    u.add_TopologyAttr("names", ALA * n_ala + SOL * n_sol)
    u.add_TopologyAttr("resnames", ["ALA"] * n_ala + ["SOL"] * n_sol)
    u.add_TopologyAttr("resids", np.arange(1, n_ala + n_sol + 1))

    box = (n_atoms / density) ** (1 / 3)
    u.dimensions = [box, box, box, 90, 90, 90]
    u.atoms.positions = np.random.default_rng(seed).uniform(0, box, (n_atoms, 3)).astype(np.float32)
    return u


def write(u, xtc_path, n_frames, step=0.1, dt=10.0, seed=1):
    rng = np.random.default_rng(seed)
    xyz = u.atoms.positions.copy()
    with mda.Writer(xtc_path, n_atoms=u.atoms.n_atoms) as w:
        for i in range(n_frames):
            if i:
                xyz += rng.normal(0, step, xyz.shape).astype(np.float32)
            u.atoms.positions = xyz
            u.trajectory.ts.time = dt * i
            w.write(u.atoms)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atoms", type=int, default=100_000)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--protein", type=float, default=0.1, help="fraction of atoms in the protein group")
    p.add_argument("--step", type=float, default=0.1, help="random-walk step per frame (Angstrom)")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    u = noise_universe(args.atoms, args.protein)
    name = args.name or f"noise_{u.atoms.n_atoms}a_{args.frames}f"
    xtc_path = os.path.join(args.out, f"{name}.xtc")
    u.atoms.write(os.path.join(args.out, f"{name}.gro"))

    t0 = time.perf_counter()
    write(u, xtc_path, args.frames, step=args.step)
    print(f"{name}: {u.atoms.n_atoms:,d} atoms x {args.frames:,d} frames -> {xtc_path} "
          f"({os.path.getsize(xtc_path) / 2**30:.2f} GiB, {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
