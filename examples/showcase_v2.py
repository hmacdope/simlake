# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # simlake
#
# A versioned data lake for MD trajectories on Zarr + Icechunk. MDAnalysis and xarray
# can both read it, from local disk or object storage.

# %% tags=["setup"]
# Setup: two example runs from MDAnalysisTests (adenylate kinase in water), with the
# protein made whole, written into a scratch directory. run1 is the last 6 frames.
import shutil, tempfile, warnings  # noqa: E401
warnings.filterwarnings("ignore")

import MDAnalysis as mda
from MDAnalysis import transformations
from MDAnalysisTests.datafiles import TPR, XTC

workdir = tempfile.mkdtemp()
u = mda.Universe(TPR, XTC)
u.trajectory.add_transformations(transformations.unwrap(u.select_atoms("protein")))
for name, frames in (("run0", slice(None)), ("run1", slice(4, None))):
    with mda.Writer(f"{workdir}/{name}.xtc", u.atoms.n_atoms) as w:
        for ts in u.trajectory[frames]:
            w.write(u.atoms)

# %% [markdown]
# ## Create a lake and a system
#
# A **system** is one topology. Trajectories of it are **replicas**.

# %%
import simlake as sl

lake = sl.Lake.create(f"{workdir}/kinase.lake")
lake.create_system("adk", TPR, metadata={"forcefield": "oplsaa"})

# %% [markdown]
# A system's atoms are split into named **groups**, each stored so it can be read on
# its own. By default they are split by molecule type:

# %%
import pandas as pd

pd.DataFrame(lake.system("adk").groups)

# %% [markdown]
# Or pass your own as MDAnalysis selections. Groups may not overlap; atoms that no group
# claims go into `other`.

# %%
lake.create_system("adk_by_part", TPR, groups={"backbone": "backbone",
                                               "sidechains": "protein and not backbone"})

# %% [markdown]
# ## Add trajectories
#
# Reference an existing file in place (nothing is copied), or copy it in, compressed,
# with each atom group stored separately.

# %%
lake.register("adk", f"{workdir}/run0.xtc", metadata={"temperature": 300})
lake.add_replica("adk", mda.Universe(TPR, f"{workdir}/run1.xtc"), metadata={"temperature": 310});

# %%
lake.system("adk").stored_bytes(replica=1)  # bytes per group: water dominates

# %% [markdown]
# Underneath it is plain Zarr: each group is an ordinary, sharded array that any Zarr
# reader can open.

# %%
import zarr

positions = zarr.open_group(lake.session().store, mode="r")["systems/adk/replicas/1/positions"]
for name, arr in positions.arrays():
    print(f"{name:8s} shape={arr.shape}  chunks={arr.chunks}  shards={arr.shards}")

# %% [markdown]
# ## Find data

# %%
lake.query("SELECT system, replica, storage, n_frames, temperature FROM trajectories")

# %% [markdown]
# ## Analyse with MDAnalysis
#
# A replica is a normal `Universe`. In-place XTC replicas read about 2x faster than
# MDAnalysis' own XTC reader (see the benchmarks in the README).

# %%
from MDAnalysis.analysis import rms

u = lake.universe("adk", replica=0)
rms.RMSD(u, select="backbone").run().results.rmsd[:, 2].round(2)

# %%
protein = lake.universe("adk", replica=1, groups="protein")  # reads only the protein's chunks
protein

# %%
mda.Universe("simlake://" + lake.location + "?system=adk&replica=1")  # or by URI

# %% [markdown]
# ## Analyse with xarray + dask
#
# `open_dataset` puts every replica in one xarray Dataset, padded to the longest along
# `frame`. No coordinates are read: every variable is a **dask array**, a grid of chunks
# that maps onto the chunks stored in the lake.

# %%
ds = lake.open_dataset("adk")
ds

# %% [markdown]
# ### 1. The dask array behind a variable
#
# `.data` is the dask array. Each block is one stored chunk (or a batch of XTC frames
# for in-place replicas), read only when something needs it.

# %%
pos = ds.position
pos.data

# %%
print("chunks along each dim:", dict(zip(pos.dims, pos.chunks)))
print("groups:", ds.md.groups, "-> one stored array per group, so no block mixes groups")
print(f"blocks: {pos.data.npartitions} | size if loaded: {pos.nbytes / 2**20:.1f} MB | read so far: 0 bytes")

# %% [markdown]
# ### 2. Build computations: still lazy
#
# `.md.group()` picks whole atom groups, and `.md.select()` takes any MDAnalysis
# selection; both are an `isel` on the `atom` dimension, and only the chunks holding
# those atoms will be read. Arithmetic on the result only extends the dask task graph.

# %%
ds.md.group("protein").position.data  # one block along atom: the protein's chunks only

# %%
ca = ds.md.select("name CA").position            # (replica, frame, atom, spatial)
ca

# %%
rmsf = ((ca - ca.mean("frame")) ** 2).sum("spatial").mean("frame") ** 0.5
rmsd = ds.md.select("backbone").md.rmsd()        # MDAnalysis' QCP kernel, per frame, via xr.apply_ufunc

print(type(rmsf.data).__name__, "with", len(rmsf.data.__dask_graph__()), "tasks")
print(type(rmsd.data).__name__, "with", len(rmsd.data.__dask_graph__()), "tasks")

# %% [markdown]
# ### 3. Your own per-frame function
#
# `xr.apply_ufunc(..., dask="parallelized")` runs a plain NumPy function on every block.
# The core dimensions (`atom`, `spatial`) reach the function whole, and dask
# handles `replica` and `frame`.

# %%
import numpy as np
import xarray as xr


def radius_of_gyration(xyz):
    """xyz: (..., atom, 3) NumPy array -> (...,)"""
    centred = xyz - xyz.mean(axis=-2, keepdims=True)
    return np.sqrt((centred ** 2).sum(axis=-1).mean(axis=-1))


rg = xr.apply_ufunc(
    radius_of_gyration, ca,
    input_core_dims=[["atom", "spatial"]],
    dask="parallelized", output_dtypes=[ca.dtype],
).rename("rg")
rg

# %% [markdown]
# ### 4. Run: `dask.compute` with a chosen scheduler
#
# Computing several results in one `dask.compute` call shares the chunk reads between
# them. The threaded scheduler is the simplest choice: no pickling and no forking.

# %%
import dask
from dask.diagnostics import ProgressBar

with dask.config.set(scheduler="threads", num_workers=8), ProgressBar():
    rmsf, rmsd, rg = dask.compute(rmsf, rmsd, rg)

rmsd.round(2)

# %% [markdown]
# ### 5. Scale out with `dask.distributed`
#
# The same code runs on a cluster. `LocalCluster` starts worker processes on this
# machine. On HPC, swap in `dask_jobqueue.SLURMCluster`, and in the cloud, Coiled or
# Kubernetes. Workers open the lake themselves from the chunk references in the graph.

# %%
from dask.distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=2, threads_per_worker=2, dashboard_address=":0")
client = Client(cluster)
print("dashboard (live while the cluster runs):", client.dashboard_link)

# %%
backbone = ds.md.select("backbone").position.persist()  # read once, keep in worker memory
rmsd_on_cluster = ds.md.select("backbone").md.rmsd().compute()  # .compute() now runs on the cluster
print("persisted blocks:", backbone.data.npartitions,
      "| same as threaded result:", bool(np.allclose(rmsd_on_cluster, rmsd, equal_nan=True)))

# %%
client.close()
cluster.close()

# %% [markdown]
# Results come back as labelled xarray objects:

# %%
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlelocation": "left", "axes.grid": True, "grid.color": "#e4e3df",
                     "axes.axisbelow": True, "legend.frameon": False})
fig, ax = plt.subplots(figsize=(7, 2.6))
for rep, color in zip(rmsf.replica.values, ["#2a78d6", "#eb6834"]):
    ax.plot(ca.atom_resid, rmsf.sel(replica=rep), color=color, lw=1.5, label=f"replica {rep}")
ax.set(xlabel="residue", ylabel="Cα RMSF (Å)", title="Per-residue flexibility, all replicas")
ax.legend();

# %% [markdown]
# ## At scale: a real membrane protein
#
# The example files above have 10 frames, too few for parallelism to show. Here is
# YiiP, a zinc transporter in a lipid membrane, from
# [MDAnalysisData](https://www.mdanalysis.org/MDAnalysisData/yiip_equilibrium.html):
# 111,815 atoms x 9,001 frames (90 ns), a 3.5 GB XTC. The copy in `data/` is used when
# it exists; otherwise MDAnalysisData downloads it (once).

# %%
import os
import time

from MDAnalysisData import yiip_equilibrium

data_home = next((d for d in ("data/mdanalysisdata", "../data/mdanalysisdata") if os.path.isdir(d)), None)
yiip = yiip_equilibrium.fetch_yiip_equilibrium_long(data_home=data_home)

lake.create_system("yiip", yiip.topology, metadata={"source": "MDAnalysisData"})
pd.DataFrame(lake.system("yiip").groups)  # default groups: a real membrane system

# %% [markdown]
# Register the XTC in place (it only reads frame headers), then copy it in as well, so
# both kinds of replica are there to compare.

# %%
t0 = time.perf_counter()
lake.register("yiip", yiip.trajectory)                          # replica 0: in place
print(f"register in place: {time.perf_counter() - t0:.1f} s")

t0 = time.perf_counter()
lake.add_replica("yiip", mda.Universe(yiip.topology, yiip.trajectory))  # replica 1: native
print(f"copy in:           {time.perf_counter() - t0:.0f} s")

big = lake.open_dataset("yiip")
print(f"{big.position.nbytes / 2**30:.1f} GiB of coordinates as {big.position.data.npartitions:,d} dask blocks")

# %% [markdown]
# ### Watch dask work through it
#
# Start a `dask.distributed` client and open its **dashboard**: live task stream, worker
# CPU and memory, and progress per task type. The workers are threads in this process
# (`processes=False`), which avoids reopening the lake in every task.

# %%
import webbrowser
from dask.distributed import Client, LocalCluster, progress

cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=8, dashboard_address=":0")
client = Client(cluster)
print("dashboard:", client.dashboard_link)
if not os.environ.get("SIMLAKE_HEADLESS"):  # set when the notebook is regenerated in batch
    webbrowser.open(client.dashboard_link)
client

# %% [markdown]
# Per-atom fluctuation for **every** atom in both replicas: one pass over all the
# coordinates, about 22 GiB, as thousands of blocks. Building it reads nothing:

# %%
fluct = big.position.var("frame").sum("spatial") ** 0.5   # (replica, atom), single pass
print(f"{len(fluct.data.__dask_graph__()):,d} tasks, nothing read yet")

# %% [markdown]
# `client.compute` starts the work in the background and returns a future;
# `progress` follows it. Watch the dashboard's task stream fill up as blocks are read,
# decoded and reduced on 8 threads.

# %%
t0 = time.perf_counter()
future = client.compute(fluct)
progress(future, notebook=False)  # dask's own progress bar; text so it also shows when saved
fluct_out = future.result()
t_cluster = time.perf_counter() - t0
print(f"\n8 threads: {t_cluster:.1f} s, {2 * big.sizes['frame'] / t_cluster:,.0f} frames/s over both replicas")

# %% [markdown]
# For comparison, the same computation on a single thread:

# %%
with dask.config.set(scheduler="synchronous"):
    t0 = time.perf_counter()
    fluct.compute()
    t_single = time.perf_counter() - t0
print(f"1 thread: {t_single:.1f} s  ->  speed-up with 8 threads: {t_single / t_cluster:.1f}x")

# %% [markdown]
# The speed-up flattens beyond about 8 threads: zarr-python coordinates every read on a
# single event-loop thread, which becomes the bottleneck. Decoding (pcodec, and the
# Cython XTC decoder) already runs off that thread and releases the GIL.

# %%
by_group = fluct_out.sel(replica=1).groupby("atom_group").mean()
fig, ax = plt.subplots(figsize=(6, 2.4))
ax.barh(by_group.atom_group, by_group, color="#2a78d6", height=0.5)
for y, v in enumerate(by_group.values):
    ax.text(v, y, f"  {v:.1f} Å", va="center")
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set(xlabel="mean per-atom fluctuation over 90 ns (Å)", title="How much each group moves")
ax.set_xlim(0, float(by_group.max()) * 1.2);

# %% [markdown]
# A lighter pair of analyses on the protein only: backbone RMSD of the dimer, and
# per-residue Cα fluctuation of each chain. Only the protein group's chunks are read.

# %%
rmsd_yiip = big.isel(replica=[1]).md.select("protein and backbone").md.rmsd()
ca_yiip = big.isel(replica=[1]).md.select("protein and name CA").position
rmsf_yiip = ca_yiip.var("frame").sum("spatial") ** 0.5

futures = client.compute([rmsd_yiip, rmsf_yiip])
progress(futures, notebook=False)
rmsd_out, rmsf_out = client.gather(futures)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3))
t_ns = big.time.sel(replica=1) / 1000
ax1.plot(t_ns, rmsd_out.sel(replica=1), color="#2a78d6", lw=1)
ax1.set(xlabel="time (ns)", ylabel="backbone RMSD (Å)", title="YiiP dimer, backbone RMSD")

rmsf_native = rmsf_out.sel(replica=1)
for segid, color in (("PROA", "#2a78d6"), ("PROB", "#eb6834")):
    chain = rmsf_native.where(rmsf_native.atom_segid == segid, drop=True)
    ax2.plot(chain.atom_resid, chain, color=color, lw=1.2, label=f"chain {segid[-1]}")
ax2.set(xlabel="residue", ylabel="Cα RMSF (Å)", title="Per-residue flexibility, both chains")
ax2.legend()
plt.tight_layout();

# %%
client.close()
cluster.close()

# %% [markdown]
# ## Versions
#
# Every write is a commit. Tag what you analysed, and read it back later.

# %%
lake.tag("paper-v1")
lake.add_replica("adk", mda.Universe(TPR, f"{workdir}/run1.xtc"))

lake.system("adk").replicas(), lake.system("adk", ref="paper-v1").replicas()

# %%
with lake.transaction("two more replicas") as tx:  # all or nothing
    tx.system("adk").add_replica(mda.Universe(TPR, f"{workdir}/run1.xtc"))
    tx.system("adk").add_replica(mda.Universe(TPR, f"{workdir}/run1.xtc"))

lake.history()[["snapshot", "message"]]

# %% [markdown]
# ## Check referenced files

# %%
lake.verify()

# %% [markdown]
# ## On S3
#
# The same API, with a lake in a bucket and trajectories referenced from another bucket.

# %% tags=["setup"]
# Setup: a local moto server stands in for S3.
import logging, socket
from moto.server import ThreadedMotoServer
logging.getLogger("werkzeug").setLevel(logging.ERROR)
with socket.socket() as s:
    s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
server = ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False); server.start()
sl.configure("s3://", endpoint_url=f"http://127.0.0.1:{port}", allow_http=True,
             region="us-west-2", access_key_id="demo", secret_access_key="demo")
fs = sl.locations.filesystem("s3://")
fs.mkdir("lakes"); fs.mkdir("raw"); fs.put(f"{workdir}/run0.xtc", "raw/adk/run0.xtc");

# %%
cloud = sl.Lake.create("s3://lakes/kinase")
cloud.create_system("adk", TPR)
cloud.register("adk", "s3://raw/adk/run0.xtc")

mda.Universe(cloud.uri("adk", replica=0))

# %%
cloud.open_dataset("adk").md.select("backbone").md.rmsd().compute().round(2)

# %% tags=["setup"]
server.stop()
shutil.rmtree(workdir, ignore_errors=True)

# %%

# %%
