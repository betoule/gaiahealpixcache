import os
import tempfile
from pathlib import Path
import hashlib
import shutil
import numpy as np


def get_cache_dir(path=False):
    """Determine the appropriate cache directory based on the OS.

    Parameters
    ----------
    path : bool, optional
        If True return a pathlib.Path object instead of a string.

    Returns
    -------
    str or Path
        Path to the gaiahealpixcache cache directory.
    """
    if os.name == "nt":
        cache_dir = Path(os.getenv("LOCALAPPDATA")) / "Cache" / "gaiahealpixcache"
    elif os.name == "posix":
        if "XDG_CACHE_HOME" in os.environ:
            cache_dir = Path(os.environ["XDG_CACHE_HOME"]) / "gaiahealpixcache"
        else:
            cache_dir = Path.home() / ".cache" / "gaiahealpixcache"
    else:
        raise OSError("Unsupported operating system")

    if path:
        return cache_dir
    return str(cache_dir)


def cached_download(url):
    """Download a file from the web with caching.

    Downloads to a temporary file first, then atomically moves to the
    final cache path. This prevents corrupted partial files from being
    cached when a download fails.

    Parameters
    ----------
    url : str
        The URL to download from.

    Returns
    -------
    str
        Path to the cached file.
    """
    cache_dir = get_cache_dir()

    os.makedirs(cache_dir, exist_ok=True)

    filename = hashlib.md5(url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path):
        return cache_path

    import requests

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=cache_dir, prefix=f".{filename}.", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "wb") as file:
            from tqdm import tqdm

            with tqdm(
                total=total, unit="B", unit_scale=True, desc=os.path.basename(url)
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    pbar.update(len(chunk))

        os.replace(tmp_path, cache_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return cache_path


def clear_cache():
    """Clear the cache directory used by cached_download.

    Returns
    -------
    None
    """
    cache_dir = get_cache_dir()

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    else:
        print(f"Cache directory {cache_dir} does not exist.")


def join(*args, **keys):
    """Join two or more np.recarray of same size.

    Single column regular arrays can be provided as key arguments,
    creating a column with a name corresponding to the key.

    Parameters
    ----------
    *args : np.recarray
        Recarrays to join, all must have the same length.
    **keys : dict
        Named arrays to add as columns.

    Returns
    -------
    np.recarray
        Joined recarray with all columns from all input arrays.
    """
    return np.rec.fromarrays(
        [nt[k] for nt in args for k in nt.dtype.names] + [keys[k] for k in keys],
        names=[k for nt in args for k in nt.dtype.names] + list(keys),
    )
