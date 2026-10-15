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
# atoms are split into groups, by default by molecule type, or your own:
#   lake.create_system("adk", TPR, groups={"protein": "protein", "ligand": "resname ATP", ...})

# %% [markdown]
# ## Add trajectories
#
# Reference an existing file in place (nothing is copied), or copy it in, compressed,
# with each atom group stored separately.

# %%
lake.register("adk", f"{workdir}/run0.xtc", metadata={"temperature": 300})
lake.add_replica("adk", mda.Universe(TPR, f"{workdir}/run1.xtc"), metadata={"temperature": 310});

# %% [markdown]
# ## Find data

# %%
lake.query("SELECT system, replica, storage, n_frames, temperature FROM trajectories")

# %% [markdown]
# ## Analyse with MDAnalysis
#
# A replica is a normal `Universe`.

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
print("groups:", ds.md.groups, "-> atom chunk edges sit on group boundaries")
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
