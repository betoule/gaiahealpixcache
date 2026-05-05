"""Collection of tools to handle computation and download cache in gaiahealpixcache"""

import os
from pathlib import Path
import hashlib
import shutil
import pickle
import numpy as np

def get_cache_dir(jit=False, path=False):
    """Determine the appropriate cache directory based on the OS.

    parameters:
    jit: bool, if True return the path to the jit cache subdirectory.
    path: bool, if True return a pathlib.Path object instead of a string
    return: None
    """

    if os.name == "nt":  # Windows
        cache_dir = Path(os.getenv("LOCALAPPDATA")) / "Cache" / "gaiahealpixcache"
    elif os.name == "posix":  # Unix-like systems
        if "XDG_CACHE_HOME" in os.environ:
            cache_dir = Path(os.environ["XDG_CACHE_HOME"]) / "gaiahealpixcache"
        else:
            cache_dir = Path.home() / ".cache" / "gaiahealpixcache"
    else:
        raise OSError("Unsupported operating system")
    if jit:
        cache_dir = cache_dir / "jit"
    if path:
        return cache_dir
    return str(cache_dir)


def cached_download(url):
    """
    Download a file from the web with caching.

    :param url: The URL to download from.
    :return: Path to the cached file.
    """
    cache_dir = get_cache_dir()

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Use URL hash for filename to avoid conflicts
    filename = hashlib.md5(url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path):
        print(f"Using cached file: {cache_path}")
        return cache_path

    # Download the file
    # this module is a bit slow to import (don't unless needed)
    import requests  # pylint: disable=import-outside-toplevel

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    with open(cache_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"File downloaded and cached at: {cache_path}")
    return cache_path

def clear_cache(jit=False):
    """
    Clear the cache directory used by cached_download.

    :param jit: Optional clear only the jit subdirectoryfor cache.
    :return: None
    """
    cache_dir = get_cache_dir(jit)

    if os.path.exists(cache_dir):
        # Slow import only if needed
        import jax  # pylint: disable=import-outside-toplevel

        shutil.rmtree(cache_dir)  # Remove the entire directory
        jax.experimental.compilation_cache.compilation_cache.reset_cache()
        print(f"Cache directory {cache_dir} has been cleared.")
    else:
        print(f"Cache directory {cache_dir} does not exist.")


