"""Collection of tools to handle computation and download cache in polyopticx"""

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
        cache_dir = Path(os.getenv("LOCALAPPDATA")) / "Cache" / "polyopticx"
    elif os.name == "posix":  # Unix-like systems
        if "XDG_CACHE_HOME" in os.environ:
            cache_dir = Path(os.environ["XDG_CACHE_HOME"]) / "polyopticx"
        else:
            cache_dir = Path.home() / ".cache" / "polyopticx"
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

def join(*args, **keys):
    '''Join two or more np.recarray of same size.

    Single column regular arrays can be provided as key arguments,
    creating a column with a name corresponding to the key.

    '''
    return np.rec.fromarrays([nt[k] for nt in args for k in nt.dtype.names]+[keys[k] for k in keys], names=[k for nt in args for k in nt.dtype.names] + [k for k in keys])

def sympy_cached(constant_sympy_computation):
    """Decorator to cache sympy expressions from functions that always
    returns the same expression.

    This decorator is designed for functions that consistently return
    an identical sympy expression of a specified set of symbols, and
    which need expensive symbolic computations at their creation. It
    stores the result in a file-based cache using pickle
    serialization, loading from the cache if available or computing
    and saving it otherwise. The cache file is named based on the
    function’s name and stored in a directory returned by
    `get_cache_dir()`.

    Parameters
    ----------
    constant_sympy_computation : callable
        A function of sympy symbols that always returns the same expression.
        Typically used for expensive-to-compute relations.

    Returns
    -------
    callable
        A wrapped function that returns the cached relation subs for the provided
        symbols, or computes, caches, and returns it if not.

    Notes
    -----
    - The cache is stored in a file named 'func_cache_<function_name>' within the
      directory specified by `get_cache_dir()`.
    - The decorator assumes `constant_sympy_computation` is deterministic and side-effect-free.
    - It correctly performs the substitution of variables between the first and subsequent calls.
    - Uses `pickle` for serialization, so the returned object must be picklable.

    Examples
    --------
    >>> @sympy_cached
    ... def square(x):
    ...     return x**2  # Expensive computation (bad example)
    >>> x, y = sympy.symbols('x y')
    >>> result = square(x)  # Computes and caches
    >>> result2 = square(y)  # Loads from cache

    """
    cache_dir = get_cache_dir()

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # Method, we keep the class name
        cache_path = os.path.join(
            cache_dir,
            f"func_cache_{constant_sympy_computation.__self__}"
            f"_{constant_sympy_computation.__name__}",
        )
    except AttributeError:
        cache_path = os.path.join(
            cache_dir, f"func_cache_{constant_sympy_computation.__name__}"
        )

    def cached_function(*args):
        if os.path.exists(cache_path):
            print(f"Using cached file: {cache_path}")
            with open(cache_path, "rb") as fid:
                result = pickle.load(fid)
                oldargs = pickle.load(fid)
            try:
                return result.subs(dict(zip(oldargs, args)))
            except AttributeError:
                return tuple(r.subs(dict(zip(oldargs, args))) for r in result)
        else:
            result = constant_sympy_computation(*args)
            with open(cache_path, "wb") as fid:
                pickle.dump(result, fid)
                pickle.dump(args, fid)
            return result

    return cached_function

def save(filename, obj):
    with open(filename, 'wb') as fid:
        pickle.dump(obj, fid)

def load(filename):
    with open(filename, 'rb') as fid:
        return pickle.load(fid)

