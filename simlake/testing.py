"""Synthetic trajectories for demos, tests and benchmarks.

``noise_trajectory`` writes a random-walk XTC of any size with a minimal topology:
the first ``protein`` fraction of atoms are 10-atom ALA residues (N CA C O CB H HA HB1
HB2 HB3, so ``protein`` and ``backbone`` select), the rest 3-atom SOL waters.

Read speed on these files is representative. Compression is a worst case: there is no
local structure to exploit. Use a real trajectory for compression numbers.
"""

import os

import numpy as np

ALA = ["N", "CA", "C", "O", "CB", "H", "HA", "HB1", "HB2", "HB3"]
SOL = ["OW", "HW1", "HW2"]


def noise_universe(n_atoms, protein=0.1, density=0.1, seed=0):
    """A Universe with an ALA "protein" and SOL "water", atoms placed uniformly in a box."""
    import MDAnalysis as mda

    n_protein = int(n_atoms * protein) // len(ALA) * len(ALA)
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


def noise_trajectory(directory, n_atoms, n_frames, protein=0.1, step=0.1, dt=10.0, seed=0, name=None):
    """Write ``<name>.gro`` and ``<name>.xtc`` (random walk, ``step`` A per frame).

    Returns ``(topology_path, trajectory_path)``.
    """
    import MDAnalysis as mda

    os.makedirs(directory, exist_ok=True)
    u = noise_universe(n_atoms, protein, seed=seed)
    name = name or f"noise_{u.atoms.n_atoms}a_{n_frames}f"
    gro = os.path.join(directory, f"{name}.gro")
    xtc = os.path.join(directory, f"{name}.xtc")
    u.atoms.write(gro)

    rng = np.random.default_rng(seed + 1)
    xyz = u.atoms.positions.copy()
    with mda.Writer(xtc, n_atoms=u.atoms.n_atoms) as w:
        for i in range(n_frames):
            if i:
                xyz += rng.normal(0, step, xyz.shape).astype(np.float32)
            u.atoms.positions = xyz
            u.trajectory.ts.time = dt * i
            w.write(u.atoms)
    return gro, xtc
