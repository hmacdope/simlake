"""Make large synthetic trajectories for benchmarking (see simlake.testing).

Random-walk XTCs of any size with a minimal ALA "protein" + SOL "water" topology. Read
speed on these is representative; compression is a worst case (no local structure).

Run, for example:

    python benchmarks/make_trajectories.py --atoms 100000 --frames 5000
    python benchmarks/make_trajectories.py --atoms 1000000 --frames 500
"""

import argparse
import os
import time
import warnings

from simlake.testing import noise_trajectory

warnings.filterwarnings("ignore")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atoms", type=int, default=100_000)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--protein", type=float, default=0.1, help="fraction of atoms in the protein group")
    p.add_argument("--step", type=float, default=0.1, help="random-walk step per frame (Angstrom)")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    t0 = time.perf_counter()
    _, xtc = noise_trajectory(args.out, args.atoms, args.frames, protein=args.protein,
                              step=args.step, name=args.name)
    print(f"{os.path.basename(xtc)}: {args.frames:,d} frames -> {xtc} "
          f"({os.path.getsize(xtc) / 2**30:.2f} GiB, {time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
