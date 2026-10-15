"""Lakes and source files on S3, against a local moto server."""

import logging
import multiprocessing
import socket
import uuid

import dask
import numpy as np
import pytest
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from MDAnalysisTests.datafiles import TPR, XTC

import simlake as sl
from simlake import locations

moto_server = pytest.importorskip("moto.server")


@pytest.fixture(scope="module")
def s3():
    logging.getLogger("werkzeug").setLevel(logging.ERROR)  # moto request log
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    server.start()
    endpoint = f"http://127.0.0.1:{port}"
    sl.configure("s3://", endpoint_url=endpoint, allow_http=True, region="us-west-2",
                 access_key_id="testing", secret_access_key="testing")
    fs = locations.filesystem("s3://lake")
    fs.mkdir("lake")
    fs.mkdir("raw")
    fs.put(XTC, "raw/adk/adk_water.xtc")
    yield fs
    server.stop()
    locations._CONFIG.clear()


@pytest.fixture
def lake(s3):
    lake = sl.Lake.create(f"s3://lake/{uuid.uuid4().hex}")
    lake.create_system("adk_water", TPR, metadata={"forcefield": "oplsaa"})
    return lake


def test_native_replica_on_s3(lake):
    lake.add_replica("adk_water", mda.Universe(TPR, XTC))
    reopened = sl.Lake.open(lake.location)
    u = reopened.universe("adk_water", 0)
    o = mda.Universe(TPR, XTC)
    assert u.trajectory.n_frames == 10
    for _ in zip(u.trajectory, o.trajectory):
        assert np.abs(u.atoms.positions - o.atoms.positions).max() <= 0.0051
    assert reopened.catalog().storage.tolist() == ["native"]


def test_uri_names_the_bucket(lake):
    lake.add_replica("adk_water", mda.Universe(TPR, XTC))
    uri = lake.uri("adk_water", 0)
    assert uri.startswith("simlake://s3://lake/")
    assert mda.Universe(uri).trajectory.n_frames == 10


def test_virtual_xtc_in_another_bucket(lake, s3):
    lake.register("adk_water", "s3://raw/adk/adk_water.xtc")
    src = lake.system("adk_water").replica_attrs(0)["sources"][0]
    assert src["uri"] == "s3://raw/adk/adk_water.xtc" and src["etag"]

    u = sl.Lake.open(lake.location).universe("adk_water", 0)
    o = mda.Universe(TPR, XTC)
    for _ in zip(u.trajectory, o.trajectory):
        np.testing.assert_array_equal(u.atoms.positions, o.atoms.positions)


def test_dask_analysis_on_s3(lake):
    lake.register("adk_water", "s3://raw/adk/adk_water.xtc")
    lake.add_replica("adk_water", mda.Universe(TPR, XTC))
    ds = lake.open_dataset("adk_water")
    with dask.config.set(scheduler="threads"):
        got = ds.md.select("backbone").md.rmsd().compute()
    expected = rms.RMSD(mda.Universe(TPR, XTC), select="backbone").run().results.rmsd[:, 2]
    np.testing.assert_allclose(got.sel(replica=0), expected, atol=1e-5)  # virtual: exact source
    np.testing.assert_allclose(got.sel(replica=1), expected, atol=0.01)  # native: quantized


def test_spawned_workers_inherit_endpoint(lake):
    """Workers started with spawn/forkserver get the S3 configuration with the reader."""
    lake.add_replica("adk_water", mda.Universe(TPR, XTC))
    u = lake.universe("adk_water", 0)
    serial = rms.RMSD(u, select="backbone").run().results.rmsd
    previous = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method("spawn", force=True)
    try:
        parallel = rms.RMSD(u, select="backbone").run(backend="multiprocessing", n_workers=2).results.rmsd
    finally:
        multiprocessing.set_start_method(previous, force=True)
    np.testing.assert_allclose(serial, parallel)


def test_changed_object_detected(lake, s3):
    s3.put(XTC, "raw/adk/changing.xtc")
    lake.register("adk_water", "s3://raw/adk/changing.xtc")
    assert lake.verify().status.tolist() == ["ok"]
    s3.pipe("raw/adk/changing.xtc", b"\x00" * 1024)  # overwrite: new ETag
    assert lake.verify().status.tolist() == ["changed"]
    with pytest.raises(Exception, match="checksum"):
        lake.universe("adk_water", 0)
    s3.rm("raw/adk/changing.xtc")
    assert lake.verify().status.tolist() == ["missing"]
