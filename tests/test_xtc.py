import os
import shutil

import dask
import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from MDAnalysisTests.datafiles import TPR, XTC

import simlake as sl
from simlake import xtc

DATA = os.path.dirname(XTC)
TEST_XTCS = sorted(f for f in os.listdir(DATA) if f.endswith(".xtc"))


def frames_of(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    info = xtc.scan(open(path, "rb"), len(raw))
    return info, [raw[o:o + n] for o, n in zip(info["offset"], info["length"])]


@pytest.mark.parametrize("name", TEST_XTCS)
def test_decoder_bit_exact_against_mdanalysis(name):
    path = os.path.join(DATA, name)
    info, frames = frames_of(path)
    reader = mda.coordinates.XTC.XTCReader(path, convert_units=False)
    assert info["n_atoms"] == reader.n_atoms
    assert len(frames) == reader.n_frames
    for raw, ts in zip(frames, reader):
        np.testing.assert_array_equal(xtc.decode_frame(raw), ts.positions)
        assert info["step"][ts.frame] == ts.data["step"]
        assert info["time"][ts.frame] == pytest.approx(ts.time)


def test_decoder_uncompressed_small_frames(tmp_path):
    """XTC stores frames of <= 9 atoms uncompressed."""
    path = str(tmp_path / "small.xtc")
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 30, size=(4, 5, 3)).astype(np.float32)
    u = mda.Universe.empty(5, trajectory=True)
    with mda.Writer(path, n_atoms=5) as w:
        for x in coords:
            u.atoms.positions = x
            w.write(u)
    reader = mda.coordinates.XTC.XTCReader(path, convert_units=False)
    _, frames = frames_of(path)
    for raw, ts in zip(frames, reader):
        np.testing.assert_array_equal(xtc.decode_frame(raw), ts.positions)


@pytest.fixture
def xtc_copy(tmp_path):
    d = tmp_path / "sources"
    d.mkdir()
    path = d / "adk_water.xtc"
    shutil.copy(XTC, path)
    return str(path)


@pytest.fixture
def lake(tmp_path, xtc_copy):
    lake = sl.Lake.create(tmp_path / "lake")
    lake.create_system("adk_water", TPR)
    lake.register("adk_water", xtc_copy)
    return lake


def test_virtual_xtc_matches_mdanalysis_exactly(lake, xtc_copy):
    assert lake.catalog().storage.tolist() == ["virtual:xtc"]
    v = lake.universe("adk_water", 0)
    o = mda.Universe(TPR, xtc_copy)
    assert v.trajectory.n_frames == o.trajectory.n_frames
    assert v.trajectory.dt == pytest.approx(o.trajectory.dt)
    for _ in zip(v.trajectory, o.trajectory):
        np.testing.assert_array_equal(v.atoms.positions, o.atoms.positions)
        assert v.trajectory.time == pytest.approx(o.trajectory.time)
        assert v.trajectory.ts.data["step"] == o.trajectory.ts.data["step"]
        np.testing.assert_allclose(v.dimensions, o.dimensions, rtol=1e-6)


def test_virtual_xtc_random_access_and_groups(lake, xtc_copy):
    o = mda.Universe(TPR, xtc_copy)
    protein = lake.universe("adk_water", 0, groups="protein")
    assert protein.atoms.n_atoms == 3341
    for frame in (7, 2, 9, 0):
        protein.trajectory[frame]
        o.trajectory[frame]
        np.testing.assert_array_equal(protein.atoms.positions, o.atoms.positions[:3341])


def write_frames(path, frames):
    """Write the given frames of the test XTC (keeping their time and step) to a new XTC."""
    u = mda.Universe(TPR, XTC)
    with mda.Writer(str(path), n_atoms=u.atoms.n_atoms) as w:
        for ts in u.trajectory[frames]:
            w.write(u.atoms)
    return str(path)


def test_writer_keeps_steps(tmp_path):
    path = write_frames(tmp_path / "part.xtc", slice(5, 10))
    with open(path, "rb") as fh:
        steps = xtc.scan(fh, os.path.getsize(path))["step"]
    original = [ts.data["step"] for ts in mda.Universe(TPR, XTC).trajectory[5:10]]
    assert list(steps) == original


def test_join_drops_duplicated_boundary_frame(tmp_path):
    """GROMACS restarts without -append repeat the boundary frame; it is kept once."""
    part1 = write_frames(tmp_path / "part1.xtc", slice(0, 6))  # frames 0-5
    part2 = write_frames(tmp_path / "part2.xtc", slice(5, 10))  # frames 5-9
    lake = sl.Lake.create(tmp_path / "lake")
    lake.create_system("adk", TPR)
    lake.register("adk", [part1, part2])

    u = lake.universe("adk", 0)
    o = mda.Universe(TPR, XTC)
    assert u.trajectory.n_frames == 10
    for _ in zip(u.trajectory, o.trajectory):
        assert u.trajectory.ts.data["step"] == o.trajectory.ts.data["step"]
        np.testing.assert_array_equal(u.atoms.positions, o.atoms.positions)
    assert [s["dropped_frames"] for s in lake.system("adk").replica_attrs(0)["sources"]] == [1, 0]


def test_join_restart_from_older_checkpoint(tmp_path):
    """A rerun from an older checkpoint replaces the frames the first run wrote after it."""
    part1 = write_frames(tmp_path / "part1.xtc", slice(0, 8))  # frames 0-7
    part2 = write_frames(tmp_path / "part2.xtc", slice(4, 10))  # restarted at frame 4
    lake = sl.Lake.create(tmp_path / "lake")
    lake.create_system("adk", TPR)
    lake.register("adk", [part1, part2])

    steps = lake.system("adk").replica_group(0)["step"][:]
    assert list(steps) == [ts.data["step"] for ts in mda.Universe(TPR, XTC).trajectory]
    assert [s["dropped_frames"] for s in lake.system("adk").replica_attrs(0)["sources"]] == [4, 0]
    u = lake.universe("adk", 0)
    u.trajectory[6]  # comes from part2
    o = mda.Universe(TPR, XTC)
    o.trajectory[6]
    np.testing.assert_array_equal(u.atoms.positions, o.atoms.positions)


def test_misaligned_replicas_warn(tmp_path, xtc_copy):
    lake = sl.Lake.create(tmp_path / "lake")
    lake.create_system("adk", TPR)
    lake.register("adk", xtc_copy)                                     # starts at 0 ps
    lake.register("adk", write_frames(tmp_path / "late.xtc", slice(4, 10)))  # starts at 400 ps
    with pytest.warns(UserWarning, match="start time"):
        lake.open_dataset("adk")


def test_aligned_replicas_do_not_warn(lake, xtc_copy):
    import warnings

    lake.register("adk_water", xtc_copy)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        lake.open_dataset("adk_water")


def test_virtual_xtc_xarray_and_dask_rmsd(lake, xtc_copy):
    ds = lake.open_dataset("adk_water")
    assert ds.storage.values.tolist() == ["virtual:xtc"]
    o = mda.Universe(TPR, xtc_copy)
    o.trajectory[4]
    np.testing.assert_array_equal(ds.md.group("protein").position.sel(replica=0, frame=4).values, o.atoms.positions[:3341])

    with dask.config.set(scheduler="threads"):
        got = ds.md.select("backbone").md.rmsd().sel(replica=0).values
    expected = rms.RMSD(o, select="backbone").run().results.rmsd[:, 2]
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_virtual_xtc_change_detected(lake, xtc_copy):
    later = os.path.getmtime(xtc_copy) + 3600
    os.utime(xtc_copy, (later, later))
    assert lake.verify().status.tolist() == ["changed"]
    with pytest.raises(Exception, match="checksum"):
        lake.universe("adk_water", 0)


def test_register_rejects_wrong_system(lake, tmp_path):
    from MDAnalysisTests.datafiles import PSF

    lake.create_system("adk_vac", PSF)
    with pytest.raises(sl.TopologyMismatch, match="atom count"):
        lake.register("adk_vac", XTC)
    with pytest.raises(sl.UnsupportedVariant):
        lake.register("adk_vac", TPR)
