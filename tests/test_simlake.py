import multiprocessing
import os
import shutil

import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from MDAnalysisTests.datafiles import DCD, PSF, TPR, XTC

import simlake as sl


@pytest.fixture
def lake(tmp_path):
    return sl.Lake.create(tmp_path / "lake")


@pytest.fixture
def water_lake(lake):
    """adk in water (47681 atoms, 10 frames) stored natively as replica 0."""
    lake.create_system("adk_water", TPR, metadata={"forcefield": "oplsaa"})
    lake.add_replica("adk_water", mda.Universe(TPR, XTC), metadata={"temperature": 300})
    return lake


@pytest.fixture
def dcd_copy(tmp_path):
    """A private copy of the DCD so tests can modify it."""
    d = tmp_path / "sources"
    d.mkdir()
    path = d / "adk.dcd"
    shutil.copy(DCD, path)
    return str(path)


def frames(u):
    for ts in u.trajectory:
        yield ts.frame, u.atoms.positions.copy(), ts.time, ts.dimensions, ts.data.get("step")


# ------------------------------------------------------------------ native round trip


def test_native_positions_time_box_round_trip(water_lake):
    orig = mda.Universe(TPR, XTC)
    lu = water_lake.universe("adk_water", 0)
    assert lu.atoms.n_atoms == orig.atoms.n_atoms
    assert lu.trajectory.n_frames == orig.trajectory.n_frames
    assert lu.trajectory.dt == pytest.approx(orig.trajectory.dt)
    for (f1, x1, t1, d1, s1), (f2, x2, t2, d2, s2) in zip(frames(lu), frames(orig)):
        assert f1 == f2
        # 0.01 A quantization: error at most half a step
        assert np.abs(x1 - x2).max() <= 0.005 + 1e-4
        assert t1 == pytest.approx(t2)
        np.testing.assert_allclose(d1, d2, atol=1e-3)
        assert s1 == s2


def test_topology_round_trip(water_lake):
    orig = mda.Universe(TPR, XTC)
    lu = water_lake.universe("adk_water", 0)
    for attr in ("names", "resnames", "resids", "segids", "types"):
        np.testing.assert_array_equal(getattr(lu.atoms, attr), getattr(orig.atoms, attr))
    np.testing.assert_allclose(lu.atoms.masses, orig.atoms.masses)
    np.testing.assert_allclose(lu.atoms.charges, orig.atoms.charges)
    assert len(lu.bonds) == len(orig.bonds)
    assert lu.residues.n_residues == orig.residues.n_residues
    # the original topology file is stored with the system
    assert "adk_oplsaa.tpr" in water_lake.system("adk_water").topology_files()


def test_rmsd_matches_original(water_lake):
    orig = mda.Universe(TPR, XTC)
    lu = water_lake.universe("adk_water", 0)
    a = rms.RMSD(orig, select="backbone").run().results.rmsd
    b = rms.RMSD(lu, select="backbone").run().results.rmsd
    np.testing.assert_allclose(a, b, atol=1e-2)


def test_external_topology_with_lake_coordinates(water_lake):
    u = mda.Universe(TPR, water_lake.uri("adk_water", 0))
    assert u.trajectory.n_frames == 10


def test_default_groups(water_lake):
    system = water_lake.system("adk_water")
    assert system.group_names == ["protein", "water", "ions"]
    sizes = {g["name"]: g["stop"] - g["start"] for g in system.groups}
    assert sizes == {"protein": 3341, "water": 44336, "ions": 4}
    assert water_lake.catalog().groups[0] == "protein:3341, water:44336, ions:4"


def test_group_universe(water_lake):
    orig = mda.Universe(TPR, XTC)
    protein = orig.select_atoms("protein")
    lu = water_lake.universe("adk_water", 0, groups="protein")
    assert lu.atoms.n_atoms == protein.n_atoms == 3341
    np.testing.assert_array_equal(lu.atoms.names, protein.names)
    for _ in zip(lu.trajectory, orig.trajectory):
        assert np.abs(lu.atoms.positions - protein.positions).max() <= 0.005 + 1e-4


def test_several_groups_universe(water_lake):
    orig = mda.Universe(TPR, XTC)
    expected = orig.select_atoms("protein or resname NA+ CL-")
    lu = water_lake.universe("adk_water", 0, groups=["ions", "protein"])
    np.testing.assert_array_equal(lu.atoms.names, expected.names)
    orig.trajectory[6]
    lu.trajectory[6]
    assert np.abs(lu.atoms.positions - expected.positions).max() <= 0.005 + 1e-4


def test_custom_groups_interleaved(lake):
    """Groups that interleave in the topology: storage order differs from topology order."""
    lake.create_system("adk", TPR, groups={"ca": "name CA", "protein_rest": "protein and not name CA"})
    system = lake.system("adk")
    assert system.group_names == ["protein_rest", "ca", "other"]  # by first atom; water+ions unclaimed
    assert not np.array_equal(system.storage_order, np.arange(47681))
    lake.add_replica("adk", mda.Universe(TPR, XTC))

    orig = mda.Universe(TPR, XTC)
    lu = lake.universe("adk", 0)
    ca = lake.universe("adk", 0, groups="ca")
    assert ca.atoms.n_atoms == 214
    for _ in zip(lu.trajectory, ca.trajectory, orig.trajectory):
        assert np.abs(lu.atoms.positions - orig.atoms.positions).max() <= 0.005 + 1e-4
        assert np.abs(ca.atoms.positions - orig.select_atoms("name CA").positions).max() <= 0.005 + 1e-4

    ds = lake.open_dataset("adk")
    xyz = ds.position.sel(replica=0, frame=2).sortby("atom").values
    orig.trajectory[2]
    assert np.abs(xyz - orig.atoms.positions).max() <= 0.005 + 1e-4


def test_overlapping_groups_rejected(lake):
    with pytest.raises(ValueError, match="overlap"):
        lake.create_system("adk", TPR, groups={"protein": "protein", "backbone": "backbone"})
    with pytest.raises(ValueError, match="reserved"):
        lake.create_system("adk", TPR, groups={"other": "protein"})
    with pytest.raises(ValueError, match="selects no atoms"):
        lake.create_system("adk", TPR, groups={"dna": "nucleic"})


def test_group_view_reads_only_its_chunks(water_lake):
    ds = water_lake.open_dataset("adk_water")
    pos = ds.position.data
    assert pos.chunks[2] == (3341, 14779, 14779, 14778, 4)  # chunk edges on group boundaries
    protein = ds.md.group("protein").position.data
    assert protein.chunks[2] == (3341,)  # one stored block along atom: no water chunks
    assert protein.npartitions == pos.npartitions // 5


@pytest.fixture
def forkserver():
    """Icechunk is not fork-safe, so parallel backends need forkserver or spawn."""
    previous = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method("forkserver", force=True)
    yield
    multiprocessing.set_start_method(previous, force=True)


@pytest.mark.parametrize("backend", ["multiprocessing", "dask"])
def test_parallel_backend_matches_serial(water_lake, forkserver, backend):
    lu = water_lake.universe("adk_water", 0)
    serial = rms.RMSD(lu, select="backbone").run().results.rmsd
    parallel = rms.RMSD(lu, select="backbone").run(backend=backend, n_workers=2).results.rmsd
    np.testing.assert_allclose(serial, parallel)


def test_fork_pickling_warns(water_lake):
    import pickle

    multiprocessing.set_start_method("fork", force=True)
    try:
        with pytest.warns(RuntimeWarning, match="not fork-safe"):
            pickle.dumps(water_lake.universe("adk_water", 0).trajectory)
    finally:
        multiprocessing.set_start_method(None, force=True)


def test_topology_mismatch_rejected(water_lake):
    with pytest.raises(sl.TopologyMismatch, match="atom count"):
        water_lake.add_replica("adk_water", mda.Universe(PSF, DCD))
    # a failed write leaves no trace
    assert water_lake.system("adk_water").replicas() == [0]


def test_existing_system_rejected(water_lake):
    with pytest.raises(sl.SystemExists):
        water_lake.create_system("adk_water", TPR)


# ------------------------------------------------------------------ virtual (option C)


def test_virtual_dcd_is_exact(lake, dcd_copy):
    lake.create_system("adk_vac", PSF)
    r = lake.register_dcd("adk_vac", dcd_copy)
    assert lake.system("adk_vac").replica_attrs(r)["storage"] == "virtual:dcd"
    v = lake.universe("adk_vac", r)
    o = mda.Universe(PSF, dcd_copy)
    assert v.trajectory.n_frames == o.trajectory.n_frames == 98
    for (f1, x1, t1, _, _), (f2, x2, t2, _, _) in zip(frames(v), frames(o)):
        np.testing.assert_array_equal(x1, x2)
        assert t1 == pytest.approx(t2)


def test_virtual_dcd_survives_reopen(lake, dcd_copy):
    lake.create_system("adk_vac", PSF)
    lake.register_dcd("adk_vac", dcd_copy)
    reopened = sl.Lake.open(lake.path)
    u = reopened.universe("adk_vac", 0)
    u.trajectory[50]
    o = mda.Universe(PSF, dcd_copy)
    o.trajectory[50]
    np.testing.assert_array_equal(u.atoms.positions, o.atoms.positions)


def test_virtual_multiple_files_concatenate(lake, dcd_copy, tmp_path):
    second = str(tmp_path / "sources" / "adk2.dcd")
    shutil.copy(DCD, second)
    lake.create_system("adk_vac", PSF)
    lake.register_dcd("adk_vac", [dcd_copy, second])
    u = lake.universe("adk_vac", 0)
    assert u.trajectory.n_frames == 196
    u.trajectory[98 + 7]
    o = mda.Universe(PSF, DCD)
    o.trajectory[7]
    np.testing.assert_array_equal(u.atoms.positions, o.atoms.positions)


def test_changed_source_detected(lake, dcd_copy):
    lake.create_system("adk_vac", PSF)
    lake.register_dcd("adk_vac", dcd_copy)
    assert (lake.verify().status == "ok").all()

    later = os.path.getmtime(dcd_copy) + 3600
    os.utime(dcd_copy, (later, later))
    assert (lake.verify().status == "changed").all()
    with pytest.raises(Exception, match="checksum"):
        lake.universe("adk_vac", 0)

    os.remove(dcd_copy)
    assert (lake.verify().status == "missing").all()


def test_mixed_virtual_and_native_replicas(lake, dcd_copy):
    lake.create_system("adk_vac", PSF)
    lake.register_dcd("adk_vac", dcd_copy)
    lake.add_replica("adk_vac", mda.Universe(PSF, DCD), frames=slice(0, 40))
    cat = lake.catalog()
    assert list(cat.storage) == ["virtual:dcd", "native"]
    ds = lake.open_dataset("adk_vac")
    assert dict(ds.sizes)["frame"] == 98
    # the shorter native replica is padded with NaN after its 40 frames
    pos = ds.position.sel(replica=1)
    assert np.isnan(pos.isel(frame=60)).all()
    np.testing.assert_allclose(
        pos.isel(frame=10).values, ds.position.sel(replica=0).isel(frame=10).values, atol=0.0051
    )


# ------------------------------------------------------------------ xarray


def test_xarray_view(water_lake):
    water_lake.add_replica("adk_water", mda.Universe(TPR, XTC), frames=slice(0, 5))
    ds = water_lake.open_dataset("adk_water")
    assert dict(ds.sizes) == {
        "replica": 2, "frame": 10, "atom": 47681, "spatial": 3, "box_vector": 3,
    }
    assert ds.n_frames.values.tolist() == [10, 5]
    assert np.isnan(ds.time.sel(replica=1, frame=7))

    ca = ds.md.select("name CA")
    assert ca.sizes["atom"] == 214
    assert set(ca.atom_group.values) == {"protein"}
    assert ds.md.groups == ["protein", "water", "ions"]

    orig = mda.Universe(TPR, XTC)
    orig.trajectory[3]
    expected = orig.select_atoms("name CA").positions
    np.testing.assert_allclose(ca.position.sel(replica=0, frame=3).values, expected, atol=0.0051)


def test_xarray_ensemble_rmsf(water_lake):
    """A per-atom fluctuation over every replica, computed lazily with dask."""
    water_lake.add_replica("adk_water", mda.Universe(TPR, XTC))
    ca = water_lake.open_dataset("adk_water").md.select("name CA").position
    rmsf = np.sqrt(((ca - ca.mean("frame")) ** 2).sum("spatial").mean("frame"))
    assert rmsf.dims == ("replica", "atom")
    values = rmsf.compute()
    np.testing.assert_allclose(values.sel(replica=0), values.sel(replica=1))


# ------------------------------------------------------------------ catalog and versions


def test_catalog_and_query(water_lake, dcd_copy):
    water_lake.create_system("adk_vac", PSF)
    water_lake.register_dcd("adk_vac", dcd_copy)
    hits = water_lake.query(
        "SELECT system, replica, n_frames FROM trajectories "
        "WHERE \"system.forcefield\" = 'oplsaa' AND temperature = 300"
    )
    assert hits.to_dict("records") == [{"system": "adk_water", "replica": 0, "n_frames": 10}]


def test_tags_pin_a_snapshot(water_lake):
    water_lake.tag("v1")
    water_lake.add_replica("adk_water", mda.Universe(TPR, XTC))
    assert water_lake.system("adk_water").replicas() == [0, 1]
    assert water_lake.system("adk_water", ref="v1").replicas() == [0]
    assert water_lake.universe("adk_water", 0, ref="v1").trajectory.n_frames == 10
    messages = water_lake.history().message.tolist()
    assert messages[0].startswith("add replica")


def test_transaction_is_atomic(water_lake):
    with pytest.raises(RuntimeError):
        with water_lake.transaction("two replicas") as tx:
            tx.system("adk_water").add_replica(mda.Universe(TPR, XTC))
            raise RuntimeError("job died")
    assert water_lake.system("adk_water").replicas() == [0]

    with water_lake.transaction("two replicas") as tx:
        s = tx.system("adk_water")
        s.add_replica(mda.Universe(TPR, XTC))
        s.add_replica(mda.Universe(TPR, XTC))
    assert water_lake.system("adk_water").replicas() == [0, 1, 2]
    assert water_lake.history().message[0] == "two replicas"


def test_stored_bytes_by_group(water_lake):
    by_group = water_lake.system("adk_water").stored_bytes(0)
    assert list(by_group) == ["protein", "water", "ions"]
    assert all(v > 0 for v in by_group.values())
    assert by_group["water"] > 5 * by_group["protein"] > 0
