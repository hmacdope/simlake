"""End-to-end tour of the simlake prototype on the MDAnalysisTests data.

Run:  python examples/quickstart.py /tmp/demo-lake
"""

import multiprocessing
import sys
import warnings

import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations
from MDAnalysis.analysis import rms
from MDAnalysisTests.datafiles import DCD, PSF, TPR, XTC

import simlake as sl

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(path):
    lake = sl.Lake.create(path)

    # 1. Native storage: a solvated system, copied in (quantized, chunked by atom group)
    lake.create_system("adk_water", TPR, metadata={"forcefield": "oplsaa", "engine": "gromacs"})
    with lake.transaction("two replicas of adk in water") as tx:
        system = tx.system("adk_water")
        system.add_replica(mda.Universe(TPR, XTC), metadata={"temperature": 300})
        system.add_replica(mda.Universe(TPR, XTC), frames=slice(0, 6), metadata={"temperature": 310})
    lake.tag("import-v1")

    # 2. Virtual storage: reference an existing DCD in place, no coordinates copied
    lake.create_system("adk_vacuum", PSF, metadata={"forcefield": "charmm"})
    lake.register_dcd("adk_vacuum", DCD)

    # 3. Discover with SQL
    print(lake.query("SELECT system, replica, storage, n_frames, total_time_ns, temperature FROM trajectories"))

    # 4. Analyse with MDAnalysis: it is just a Universe. The test XTC has the protein
    #    split across the periodic boundary, so make it whole first (uses the stored bonds).
    u = lake.universe("adk_water", 0)
    u.trajectory.add_transformations(transformations.unwrap(u.select_atoms("protein")))
    rmsd = rms.RMSD(u, select="backbone").run().results.rmsd[:, 2]
    print("backbone RMSD (A):", np.round(rmsd, 2))

    # One atom group: reads only the protein's chunks
    protein = lake.universe("adk_water", 0, groups="protein")
    print(protein, "vs", u)

    # Parallel MDAnalysis backends work, with a fork-safe start method
    multiprocessing.set_start_method("forkserver", force=True)
    par = rms.RMSD(u, select="backbone").run(backend="multiprocessing", n_workers=2).results.rmsd[:, 2]
    print("parallel RMSD matches serial:", np.allclose(rmsd, par))

    # 5. Analyse with xarray: every replica at once, lazily
    ds = lake.open_dataset("adk_water")
    ca = ds.md.select("name CA").position
    rmsf = np.sqrt(((ca - ca.mean("frame")) ** 2).sum("spatial").mean("frame")).compute()
    print("CA RMSF per replica, first 5 residues:\n", rmsf.isel(atom=slice(0, 5)).round(2).values)

    # 6. Versions: the tag still shows the lake as it was
    print("replicas at import-v1:", lake.system("adk_water", ref="import-v1").replicas())
    print(lake.history()[["snapshot", "message"]])
    print(lake.verify())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo-lake")
