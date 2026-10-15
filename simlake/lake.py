"""Lake, Transaction and System: the user-facing API."""

import contextlib
import os
import warnings
from urllib.parse import quote

import icechunk
import numpy as np
import pandas as pd
import zarr
import MDAnalysis as mda

from . import groups as _groups
from . import locations, storage, topology, xtc
from .conventions import CONVENTIONS_VERSION, DEFAULT_PRECISION
from .errors import SystemExists, TopologyMismatch, UnsupportedVariant

icechunk.set_logs_filter("error")



def _authorization(prefixes):
    return icechunk.containers_credentials({p: locations.container_credentials(p) for p in prefixes})


def _as_atoms(obj):
    if isinstance(obj, (str, os.PathLike)):
        with warnings.catch_warnings():
            # topology-only files (PSF, TPR, ...) have no coordinates; that's expected here
            warnings.filterwarnings("ignore", message="No coordinate reader found")
            obj = mda.Universe(os.fspath(obj))
    return obj.atoms  # Universe.atoms, or AtomGroup.atoms (itself)


class Lake:
    """A versioned collection of MD systems and their replicas, stored in one Icechunk repo.

    ``location`` is a local path or an object-store URL (``s3://bucket/prefix``, ...);
    see :mod:`simlake.locations` for credentials and endpoints.
    """

    def __init__(self, location, repo):
        self.location = locations.normalize(location)
        self.path = self.location  # backwards-compatible alias
        self._repo = repo

    def __repr__(self):
        return f"<Lake {self.location}>"

    # ------------------------------------------------------------------ lifecycle

    @classmethod
    def create(cls, location):
        location = locations.normalize(location)
        repo = icechunk.Repository.create(locations.lake_storage(location))
        session = repo.writable_session("main")
        root = zarr.open_group(session.store, mode="a")
        root.attrs["simlake_conventions"] = CONVENTIONS_VERSION
        root.create_group("systems")
        session.commit("create lake")
        return cls(location, repo)

    @classmethod
    def open(cls, location):
        location = locations.normalize(location)
        st = locations.lake_storage(location)
        config = icechunk.Repository.fetch_config(st)
        prefixes = list(config.virtual_chunk_containers or {}) if config is not None else []
        repo = icechunk.Repository.open(st, authorize_virtual_chunk_access=_authorization(prefixes))
        return cls(location, repo)

    @classmethod
    def open_or_create(cls, location):
        try:
            return cls.open(location)
        except Exception:
            return cls.create(location)

    def _ensure_virtual_container(self, prefix):
        """Allow this lake to reference source files under ``prefix`` (a URL ending in /)."""
        config = self._repo.config
        containers = config.virtual_chunk_containers or {}
        if any(prefix.startswith(p) for p in containers):
            return
        config.set_virtual_chunk_container(locations.virtual_container(prefix))
        self._repo = icechunk.Repository.open(
            locations.lake_storage(self.location),
            config=config,
            authorize_virtual_chunk_access=_authorization(list(containers) + [prefix]),
        )
        self._repo.save_config()

    # ------------------------------------------------------------------ sessions and versions

    def session(self, ref=None):
        """Read-only session at a branch, tag or snapshot id (default: main)."""
        if ref is None or ref in self._repo.list_branches():
            return self._repo.readonly_session(branch=ref or "main")
        if ref in self._repo.list_tags():
            return self._repo.readonly_session(tag=ref)
        return self._repo.readonly_session(snapshot_id=ref)

    @contextlib.contextmanager
    def transaction(self, message, branch="main"):
        """Group writes into one atomic commit. Nothing is visible to readers until exit."""
        tx = Transaction(self, self._repo.writable_session(branch))
        yield tx
        if tx.session.has_uncommitted_changes:
            tx.snapshot_id = tx.session.commit(message)

    def tag(self, name, snapshot_id=None):
        snapshot_id = snapshot_id or self._repo.lookup_branch("main")
        self._repo.create_tag(name, snapshot_id)
        return snapshot_id

    def history(self, branch="main"):
        return pd.DataFrame(
            [{"snapshot": s.id, "written_at": s.written_at, "message": s.message}
             for s in self._repo.ancestry(branch=branch)]
        )

    # ------------------------------------------------------------------ convenience writes

    def create_system(self, name, topology_source, **kwargs):
        with self.transaction(f"create system {name}") as tx:
            tx.create_system(name, topology_source, **kwargs)
        return self.system(name)

    def add_replica(self, system, atoms, replica=None, **kwargs):
        with self.transaction(f"add replica to {system}") as tx:
            replica = tx.system(system).add_replica(atoms, replica=replica, **kwargs)
        return replica

    def register(self, system, sources, replica=None, metadata=None):
        """Reference existing trajectory file(s) as a new replica, without copying coordinates.

        ``sources`` is a path/URL or a list of them (joined along frames). Supported:
        ``.xtc`` (local or object storage) and ``.dcd`` (local).
        """
        urls = [os.fspath(sources)] if isinstance(sources, (str, os.PathLike)) else [os.fspath(u) for u in sources]
        urls = [locations.normalize(u) for u in urls]
        kinds = {os.path.splitext(u)[1].lower() for u in urls}
        if len(kinds) != 1 or kinds - {".xtc", ".dcd"}:
            raise UnsupportedVariant(f"can only register .xtc or .dcd files (one kind per replica), got {sorted(kinds)}")
        for prefix in sorted({locations.container_prefix(u) for u in urls}):
            self._ensure_virtual_container(prefix)
        fmt = kinds.pop()[1:]
        with self.transaction(f"register {len(urls)} {fmt.upper()} file(s) in {system} (virtual)") as tx:
            replica = tx.system(system).register(urls, fmt, replica=replica, metadata=metadata)
        return replica

    def register_dcd(self, system, paths, replica=None, metadata=None):
        return self.register(system, paths, replica=replica, metadata=metadata)

    def register_xtc(self, system, sources, replica=None, metadata=None):
        return self.register(system, sources, replica=replica, metadata=metadata)

    # ------------------------------------------------------------------ reads

    def systems(self, ref=None):
        root = zarr.open_group(self.session(ref).store, mode="r")
        return sorted(name for name, _ in root["systems"].groups())

    def system(self, name, ref=None):
        root = zarr.open_group(self.session(ref).store, mode="r")
        return System(self, root[f"systems/{name}"], name, ref=ref)

    def uri(self, system, replica, ref=None, groups=None):
        q = f"system={quote(system)}&replica={replica}"
        if ref:
            q += f"&ref={quote(ref)}"
        if groups:
            q += "&groups=" + ",".join(quote(g) for g in ([groups] if isinstance(groups, str) else groups))
        return f"simlake://{self.location}?{q}"

    def universe(self, system, replica, ref=None, groups=None, **kwargs):
        """An MDAnalysis Universe for one replica (topology and coordinates from the lake).

        ``groups`` (a name or list of names) restricts the Universe to those atom groups;
        native replicas then read only those groups' chunks.
        """
        uri = self.uri(system, replica, ref=ref, groups=groups)
        return mda.Universe(uri, uri, **kwargs)

    def open_dataset(self, system, ref=None):
        """Lazy xarray view of all replicas of a system, padded along ``frame``."""
        from .xr import open_system_dataset

        return open_system_dataset(self.system(system, ref=ref))

    def catalog(self, ref=None):
        """One row per replica, with system- and replica-level metadata flattened in."""
        rows = []
        for name in self.systems(ref):
            system = self.system(name, ref)
            sattrs = system.attrs
            for r in system.replicas():
                rattrs = system.replica_attrs(r)
                time = system.replica_group(r)["time"]
                t = time[[0, -1]] if time.shape[0] else np.array([np.nan, np.nan])
                rows.append({
                    "system": name,
                    "replica": r,
                    "storage": rattrs["storage"],
                    "n_atoms": sattrs["n_atoms"],
                    "groups": ", ".join(f"{g['name']}:{g['stop'] - g['start']}" for g in sattrs["groups"]),
                    "n_frames": rattrs["n_frames"],
                    "t_start_ps": float(t[0]),
                    "t_end_ps": float(t[1]),
                    "total_time_ns": float(t[1] - t[0]) / 1000.0,
                    "precision": rattrs.get("precision"),
                    "topology_hash": sattrs["topology_hash"],
                    **{f"system.{k}": v for k, v in sattrs.get("metadata", {}).items()},
                    **{k: v for k, v in rattrs.get("metadata", {}).items()},
                })
        return pd.DataFrame(rows)

    def query(self, sql, ref=None):
        """Run SQL against the catalog, exposed as the table ``trajectories``."""
        import duckdb

        con = duckdb.connect()
        con.register("trajectories", self.catalog(ref))
        return con.execute(sql).df()

    def verify(self, ref=None):
        """Check that every file referenced by a virtual replica still exists unchanged."""
        rows = []
        for name in self.systems(ref):
            system = self.system(name, ref)
            for r in system.replicas():
                for src in system.replica_attrs(r).get("sources", []):
                    rows.append({"system": name, "replica": r, "uri": src["uri"],
                                 "status": locations.status(src)})
        return pd.DataFrame(rows, columns=["system", "replica", "uri", "status"])


class Transaction:
    """Writable view of a lake inside :meth:`Lake.transaction`."""

    def __init__(self, lake, session):
        self.lake = lake
        self.session = session
        self.root = zarr.open_group(session.store, mode="a")
        self.snapshot_id = None

    def create_system(self, name, topology_source, groups=None, metadata=None):
        """Create a system from a topology file, Universe or AtomGroup.

        ``groups`` maps group names to MDAnalysis selections and partitions the atoms;
        each group is stored so it can be read on its own. Without it, atoms are grouped
        by molecule type (protein, nucleic, lipids, water, ions, other). See
        :mod:`simlake.groups`.
        """
        if name in self.root["systems"]:
            raise SystemExists(name)
        atoms = _as_atoms(topology_source)
        top = topology.extract(atoms)
        storage_order, group_meta = _groups.layout(_groups.resolve(atoms, groups))

        files = []
        if isinstance(topology_source, (str, os.PathLike)):
            files = [os.fspath(topology_source)]
        elif isinstance(getattr(atoms.universe, "filename", None), str) and os.path.exists(atoms.universe.filename):
            files = [atoms.universe.filename]

        group = self.root["systems"].create_group(name)
        topology.write(group.create_group("topology"), top, storage_order, files)
        group.create_group("replicas")
        group.attrs.update({
            "topology_hash": topology.topology_hash(top),
            "n_atoms": int(atoms.n_atoms),
            "groups": group_meta,
            "metadata": metadata or {},
        })
        return System(self.lake, group, name, writable=True, store=self.session.store)

    def system(self, name):
        return System(self.lake, self.root[f"systems/{name}"], name, writable=True, store=self.session.store)


class System:
    """One topology plus any number of replicas (trajectories of that topology)."""

    def __init__(self, lake, group, name, ref=None, writable=False, store=None):
        self.lake = lake
        self.group = group
        self.name = name
        self.ref = ref
        self.writable = writable
        self._store = store

    def __repr__(self):
        groups = ", ".join(f"{g['name']} {g['stop'] - g['start']}" for g in self.groups)
        return f"<System {self.name!r}: {self.attrs['n_atoms']} atoms ({groups}), {len(self.replicas())} replica(s)>"

    @property
    def attrs(self):
        return dict(self.group.attrs)

    @property
    def groups(self):
        """Atom groups: list of {"name", "selection", "start", "stop"} in storage order."""
        return self.attrs["groups"]

    @property
    def group_names(self):
        return [g["name"] for g in self.groups]

    @property
    def storage_order(self):
        """Topology index of each atom as stored in native position arrays."""
        return self.group["topology/storage_order"][:]

    def topology(self):
        return topology.read(self.group["topology"])

    def topology_files(self):
        return topology.source_files(self.group["topology"])

    def replicas(self):
        return sorted(int(k) for k, _ in self.group["replicas"].groups())

    def replica_group(self, replica):
        return self.group[f"replicas/{replica}"]

    def replica_attrs(self, replica):
        return dict(self.replica_group(replica).attrs)

    def stored_bytes(self, replica):
        """Bytes stored in the lake for a native replica's positions, per atom group."""
        if self.replica_attrs(replica)["storage"] != "native":
            raise ValueError("only native replicas store coordinates in the lake")
        positions = self.replica_group(replica)["positions"]
        return {name: positions[name].nbytes_stored() for name in self.group_names}

    # ------------------------------------------------------------------ writes

    def _new_replica_group(self, replica, metadata):
        if not self.writable:
            raise PermissionError("open a transaction to modify a system")
        existing = self.replicas()
        if replica is None:
            replica = max(existing) + 1 if existing else 0
        elif replica in existing:
            raise ValueError(f"replica {replica} already exists in {self.name}")
        group = self.group["replicas"].create_group(str(replica))
        group.attrs["metadata"] = metadata or {}
        return replica, group

    def add_replica(self, atoms, replica=None, frames=slice(None), precision=DEFAULT_PRECISION,
                    level=8, metadata=None):
        """Copy a trajectory into the lake (native storage). ``atoms`` must match the topology."""
        atoms = _as_atoms(atoms)
        topology.check_match(self.attrs["topology_hash"], self.attrs["n_atoms"],
                             topology.extract(atoms), stored=self.topology())
        replica, group = self._new_replica_group(replica, metadata)
        storage.write_native(group, atoms, self.storage_order, self.groups,
                             frames=frames, precision=precision, level=level)
        return replica

    def register(self, urls, fmt, replica=None, metadata=None):
        """Reference trajectory files in place. Prefer :meth:`Lake.register`, which also
        authorizes the lake to read the files' location."""
        if fmt == "dcd":
            n_atoms = storage.dcd_layout(urls[0])["n_atoms"]
        else:
            with locations.open_source(urls[0]) as fh:
                head = fh.read(xtc.HEADER_BYTES)
            n_atoms = int(np.frombuffer(head, dtype=">i4", count=2, offset=0)[1])
        if n_atoms != self.attrs["n_atoms"]:
            raise TopologyMismatch(f"atom count differs: system has {self.attrs['n_atoms']}, {fmt.upper()} has {n_atoms}")
        replica, group = self._new_replica_group(replica, metadata)
        writer = storage.write_virtual_dcd if fmt == "dcd" else storage.write_virtual_xtc
        writer(group, self._store, group.path, urls)
        return replica

    # ------------------------------------------------------------------ reads

    def universe(self, replica, groups=None, **kwargs):
        return self.lake.universe(self.name, replica, ref=self.ref, groups=groups, **kwargs)

    def to_xarray(self):
        from .xr import open_system_dataset

        return open_system_dataset(self)
