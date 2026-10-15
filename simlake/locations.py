"""Where lakes and source files live: local paths, s3://, gs:// and az:// URLs.

Credentials and endpoints are never put in URIs. They come from the environment
(the default, as for any AWS/GCP/Azure client) or from :func:`configure`::

    sl.configure("s3://my-bucket", endpoint_url="http://localhost:9000",
                 allow_http=True, access_key_id="...", secret_access_key="...")

Configuration applies to every URL starting with the given prefix (longest prefix
wins) and is carried along when readers are pickled to worker processes.

S3 is tested (against a local moto server). GCS and Azure follow the same path but are
untested in this prototype.
"""

import datetime
import math
import os
from urllib.parse import urlparse

import icechunk

_CONFIG = {}

_S3_STORE_KEYS = ("region", "endpoint_url", "allow_http", "anonymous", "force_path_style")
_S3_CRED_KEYS = ("access_key_id", "secret_access_key", "session_token", "anonymous")


def configure(prefix, **options):
    """Set storage options for every URL that starts with ``prefix``.

    S3 options: region, endpoint_url, allow_http, anonymous, force_path_style,
    access_key_id, secret_access_key, session_token. Options not given are taken
    from the environment.
    """
    _CONFIG[prefix.rstrip("/")] = dict(options)


def snapshot():
    return {k: dict(v) for k, v in _CONFIG.items()}


def restore(config):
    for k, v in config.items():
        _CONFIG.setdefault(k, v)


def options_for(url):
    best = ""
    for prefix in _CONFIG:
        if url.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return dict(_CONFIG.get(best, {}))


def scheme(url):
    s = urlparse(str(url)).scheme
    return "s3" if s in ("s3", "s3a") else "gcs" if s in ("gs", "gcs") else "az" if s in ("az", "abfs") else "file"


def normalize(location):
    """Absolute local path, or the URL unchanged (without a trailing slash)."""
    location = os.fspath(location)
    if scheme(location) == "file":
        return os.path.abspath(location[len("file://"):] if location.startswith("file://") else location)
    return location.rstrip("/")


def _bucket_prefix(url):
    u = urlparse(url)
    return u.netloc, u.path.lstrip("/")


def _s3_kwargs(opts, keys):
    out = {k: opts[k] for k in keys if k in opts}
    if "endpoint_url" in opts and "force_path_style" in keys:
        out.setdefault("force_path_style", True)
    return out


def lake_storage(location):
    """Icechunk Storage for a lake location."""
    kind = scheme(location)
    if kind == "file":
        return icechunk.local_filesystem_storage(location)
    opts = options_for(location)
    bucket, prefix = _bucket_prefix(location)
    if kind == "s3":
        has_keys = "access_key_id" in opts or opts.get("anonymous")
        return icechunk.s3_storage(
            bucket=bucket, prefix=prefix,
            **_s3_kwargs(opts, _S3_STORE_KEYS + _S3_CRED_KEYS),
            from_env=None if has_keys else True,
        )
    if kind == "gcs":
        return icechunk.gcs_storage(bucket=bucket, prefix=prefix, from_env=not opts, **opts)
    return icechunk.azure_storage(account=opts.pop("account"), container=bucket, prefix=prefix,
                                  from_env=not opts, **opts)


def container_prefix(url):
    """The virtual chunk container prefix (a "directory") that covers a source URL."""
    if scheme(url) == "file":
        return "file://" + os.path.dirname(os.path.abspath(url)).rstrip("/") + "/"
    return url.rsplit("/", 1)[0] + "/"


def source_url(path):
    """Canonical URL Icechunk stores for a source file."""
    return "file://" + os.path.abspath(path) if scheme(path) == "file" else path


def virtual_container(prefix):
    kind = scheme(prefix)
    if kind == "file":
        store = icechunk.local_filesystem_store(prefix[len("file://"):])
    elif kind == "s3":
        opts = options_for(prefix)
        store = icechunk.s3_store(s3_compatible="endpoint_url" in opts, **_s3_kwargs(opts, _S3_STORE_KEYS))
    elif kind == "gcs":
        store = icechunk.gcs_store()
    else:
        raise NotImplementedError("virtual references into Azure are not wired up yet")
    return icechunk.VirtualChunkContainer(prefix, store)


def container_credentials(prefix):
    kind = scheme(prefix)
    if kind == "file":
        return icechunk.Credentials.LocalFileSystemAccess()
    opts = options_for(prefix)
    if kind == "s3":
        keys = _s3_kwargs(opts, _S3_CRED_KEYS)
        return icechunk.s3_credentials(**keys) if keys else icechunk.s3_credentials(from_env=True)
    if kind == "gcs":
        return icechunk.gcs_credentials(from_env=True)
    return None


# --------------------------------------------------------------------------- source files


def filesystem(url):
    """An fsspec filesystem for a source URL (local, or remote with configured options)."""
    import fsspec

    kind = scheme(url)
    if kind == "file":
        return fsspec.filesystem("file")
    opts = options_for(url)
    if kind == "s3":
        client = {k: opts[k] for k in ("endpoint_url",) if k in opts}
        if "region" in opts:
            client["region_name"] = opts["region"]
        return fsspec.filesystem(
            "s3", key=opts.get("access_key_id"), secret=opts.get("secret_access_key"),
            token=opts.get("session_token"), anon=opts.get("anonymous", False),
            client_kwargs=client, skip_instance_cache=True,
        )
    return fsspec.filesystem("gcs" if kind == "gcs" else "az", **opts)


def _strip(url):
    return url[len("file://"):] if url.startswith("file://") else url


def fingerprint(url):
    """What we record about a source so later changes can be detected.

    Local files: size + mtime. Object stores: size + ETag.
    """
    info = filesystem(url).info(_strip(url))
    fp = {"uri": source_url(url), "size": int(info["size"])}
    if scheme(url) == "file":
        fp["mtime"] = os.stat(_strip(url)).st_mtime
    else:
        fp["etag"] = info.get("ETag") or info.get("etag")
    return fp


def icechunk_checksum(fp):
    """Keyword arguments for VirtualChunkSpec so Icechunk refuses changed sources."""
    if "etag" in fp and fp["etag"]:
        return {"etag_checksum": fp["etag"]}
    # Second resolution, rounded up so the current mtime always passes.
    return {"last_updated_at_checksum": datetime.datetime.fromtimestamp(
        math.ceil(fp["mtime"]) + 1, tz=datetime.timezone.utc)}


def status(fp):
    """'ok', 'changed' or 'missing' for a recorded fingerprint."""
    try:
        now = fingerprint(fp["uri"])
    except FileNotFoundError:
        return "missing"
    same = now["size"] == fp["size"] and now.get("mtime") == fp.get("mtime") and now.get("etag") == fp.get("etag")
    return "ok" if same else "changed"


def open_source(url):
    """Open a source file for reading (seekable)."""
    return filesystem(url).open(_strip(url), "rb", block_size=8 * 2**20)
