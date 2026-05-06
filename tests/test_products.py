import json
import os

import pytest

from gaiahealpixcache.products import (
    DEFAULT_PRODUCTS,
    GaiaProduct,
    _cleanup_cache,
    _get_config_dir,
    _init_registry,
    _load_user_products,
    get_product,
    list_products,
    register_product,
    unregister_product,
)

# ---------------------------------------------------------------------------
# GaiaProduct dataclass tests
# ---------------------------------------------------------------------------


def test_gaiaproduct_defaults():
    prod = GaiaProduct(
        name="test",
        url="https://example.com/",
        md5sum_file="_MD5SUM.txt",
        file_prefix="Test_",
        file_ext=".csv.gz",
    )
    assert prod.name == "test"
    assert prod.columns == []


def test_gaiaproduct_columns():
    prod = GaiaProduct(
        name="test",
        url="https://example.com/",
        md5sum_file="_MD5SUM.txt",
        file_prefix="Test_",
        file_ext=".csv.gz",
        columns=["ra", "dec"],
    )
    assert prod.columns == ["ra", "dec"]


def test_gaiaproduct_frozen():
    prod = GaiaProduct(
        name="test",
        url="https://example.com/",
        md5sum_file="_MD5SUM.txt",
        file_prefix="Test_",
        file_ext=".csv.gz",
    )
    with pytest.raises(Exception):
        prod.name = "other"


def test_gaiaproduct_to_dict():
    prod = GaiaProduct(
        name="test",
        url="https://example.com/",
        md5sum_file="_MD5SUM.txt",
        file_prefix="Test_",
        file_ext=".csv.gz",
        columns=["ra", "dec"],
    )
    d = prod.to_dict()
    assert d["name"] == "test"
    assert d["columns"] == ["ra", "dec"]
    assert d["url"] == "https://example.com/"


def test_gaiaproduct_from_dict():
    d = {
        "name": "test",
        "url": "https://example.com/",
        "md5sum_file": "_MD5SUM.txt",
        "file_prefix": "Test_",
        "file_ext": ".csv.gz",
        "columns": ["ra", "dec"],
    }
    prod = GaiaProduct.from_dict(d)
    assert prod.name == "test"
    assert prod.columns == ["ra", "dec"]


def test_gaiaproduct_roundtrip():
    prod = GaiaProduct(
        name="test",
        url="https://example.com/",
        md5sum_file="_MD5SUM.txt",
        file_prefix="Test_",
        file_ext=".csv.gz",
        columns=["ra", "dec", "parallax"],
    )
    restored = GaiaProduct.from_dict(prod.to_dict())
    assert prod == restored


def test_gaiaproduct_config_hash():
    prod_a = GaiaProduct(
        name="test",
        url="https://example.com/",
        md5sum_file="_MD5SUM.txt",
        file_prefix="Test_",
        file_ext=".csv.gz",
        columns=["ra", "dec"],
    )
    prod_b = GaiaProduct(
        name="test",
        url="https://example.com/",
        md5sum_file="_MD5SUM.txt",
        file_prefix="Test_",
        file_ext=".csv.gz",
        columns=["dec", "ra"],
    )
    prod_c = GaiaProduct(
        name="test",
        url="https://example.com/",
        md5sum_file="_MD5SUM.txt",
        file_prefix="Test_",
        file_ext=".csv.gz",
        columns=["ra", "dec", "parallax"],
    )
    assert prod_a.config_hash() == prod_b.config_hash()
    assert prod_a.config_hash() != prod_c.config_hash()
    assert len(prod_a.config_hash()) == 8


# ---------------------------------------------------------------------------
# Default products
# ---------------------------------------------------------------------------


def test_default_source_product():
    source = DEFAULT_PRODUCTS["source"]
    assert source.name == "source"
    assert "source_id" in source.columns
    assert "ra" in source.columns
    assert "dec" in source.columns
    assert len(source.columns) == 16


def test_default_source_columns_match():
    from gaiahealpixcache.products import COLUMNS_OF_INTEREST

    source = DEFAULT_PRODUCTS["source"]
    assert source.columns == COLUMNS_OF_INTEREST


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_get_product_source():
    prod = get_product("source")
    assert prod.name == "source"
    assert isinstance(prod, GaiaProduct)


def test_get_product_not_found():
    with pytest.raises(KeyError, match="not found"):
        get_product("nonexistent_product")


def test_list_products():
    products = list_products()
    assert isinstance(products, list)
    assert "source" in products
    assert products == sorted(products)


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


def test_register_product_persists(tmp_path):
    orig_config = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path)

    try:
        _init_registry.__globals__["_product_registry"] = {}
        _init_registry()

        custom = GaiaProduct(
            name="custom_test",
            url="https://example.com/custom/",
            md5sum_file="_MD5SUM.txt",
            file_prefix="Custom_",
            file_ext=".csv.gz",
            columns=["source_id", "ra", "dec"],
        )
        register_product(custom)

        products_dir = tmp_path / "gaiahealpixcache" / "products"
        assert (products_dir / "custom_test.json").exists()

        with open(products_dir / "custom_test.json") as f:
            saved = json.load(f)
        assert saved["name"] == "custom_test"
        assert saved["columns"] == ["source_id", "ra", "dec"]

        assert get_product("custom_test").name == "custom_test"
    finally:
        _init_registry.__globals__["_product_registry"] = {}
        if orig_config is not None:
            os.environ["XDG_CONFIG_HOME"] = orig_config
        elif "XDG_CONFIG_HOME" in os.environ:
            del os.environ["XDG_CONFIG_HOME"]


def test_unregister_product(tmp_path):
    orig_config = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path)

    try:
        _init_registry.__globals__["_product_registry"] = {}
        _init_registry()

        custom = GaiaProduct(
            name="to_remove",
            url="https://example.com/",
            md5sum_file="_MD5SUM.txt",
            file_prefix="Test_",
            file_ext=".csv.gz",
            columns=["ra"],
        )
        register_product(custom)
        assert get_product("to_remove").name == "to_remove"

        unregister_product("to_remove")
        with pytest.raises(KeyError):
            get_product("to_remove")

        products_dir = tmp_path / "gaiahealpixcache" / "products"
        assert not (products_dir / "to_remove.json").exists()
    finally:
        _init_registry.__globals__["_product_registry"] = {}
        if orig_config is not None:
            os.environ["XDG_CONFIG_HOME"] = orig_config
        elif "XDG_CONFIG_HOME" in os.environ:
            del os.environ["XDG_CONFIG_HOME"]


def test_unregister_product_cleans_cache(tmp_path):
    orig_config = os.environ.get("XDG_CONFIG_HOME")
    orig_cache = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
    os.environ["XDG_CACHE_HOME"] = str(tmp_path)

    cache_dir = tmp_path / "gaiahealpixcache"
    cache_dir.mkdir(parents=True)

    try:
        _init_registry.__globals__["_product_registry"] = {}
        _init_registry()

        custom = GaiaProduct(
            name="cached_prod",
            url="https://example.com/",
            md5sum_file="_MD5SUM.txt",
            file_prefix="CachedProd_",
            file_ext=".csv.gz",
            columns=["ra"],
        )
        register_product(custom)

        cfg_hash = custom.config_hash()
        stem = custom.file_prefix.rstrip("_").rstrip(".csv").rstrip(".gz")
        cache_file = cache_dir / f"{stem}_0-63_{cfg_hash}.npy"
        cache_file.write_bytes(b"\x00")
        stale_file = cache_dir / f"other_0-63_abcd1234.npy"
        stale_file.write_bytes(b"\x00")

        unregister_product("cached_prod")

        assert not cache_file.exists()
        assert stale_file.exists()
    finally:
        _init_registry.__globals__["_product_registry"] = {}
        if orig_config is not None:
            os.environ["XDG_CONFIG_HOME"] = orig_config
        elif "XDG_CONFIG_HOME" in os.environ:
            del os.environ["XDG_CONFIG_HOME"]
        if orig_cache is not None:
            os.environ["XDG_CACHE_HOME"] = orig_cache
        elif "XDG_CACHE_HOME" in os.environ:
            del os.environ["XDG_CACHE_HOME"]


def test_cannot_unregister_default():
    with pytest.raises(KeyError, match="Cannot unregister default"):
        unregister_product("source")


# ---------------------------------------------------------------------------
# Config dir
# ---------------------------------------------------------------------------


def test_get_config_dir_posix(tmp_path):
    orig = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
    try:
        d = _get_config_dir()
        assert d == tmp_path / "gaiahealpixcache"
    finally:
        if orig is not None:
            os.environ["XDG_CONFIG_HOME"] = orig
        elif "XDG_CONFIG_HOME" in os.environ:
            del os.environ["XDG_CONFIG_HOME"]


# ---------------------------------------------------------------------------
# Load user products
# ---------------------------------------------------------------------------


def test_load_user_products_valid(tmp_path):
    products_dir = tmp_path / "gaiahealpixcache" / "products"
    products_dir.mkdir(parents=True)
    prod = GaiaProduct(
        name="test_load",
        url="http://example.com/",
        md5sum_file="_MD5SUM.txt",
        file_prefix="Test_",
        file_ext=".csv.gz",
        columns=["ra", "dec"],
    )
    with open(products_dir / "test_load.json", "w") as f:
        json.dump(prod.to_dict(), f)

    orig = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
    try:
        result = _load_user_products()
        assert "test_load" in result
        assert result["test_load"].columns == ["ra", "dec"]
    finally:
        if orig is not None:
            os.environ["XDG_CONFIG_HOME"] = orig
        elif "XDG_CONFIG_HOME" in os.environ:
            del os.environ["XDG_CONFIG_HOME"]


def test_load_user_products_invalid_json(tmp_path, capsys):
    products_dir = tmp_path / "gaiahealpixcache" / "products"
    products_dir.mkdir(parents=True)
    with open(products_dir / "bad.json", "w") as f:
        f.write("not valid json{{{")

    orig = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
    try:
        result = _load_user_products()
        assert "bad" not in result
        captured = capsys.readouterr()
        assert "Warning:" in captured.out
    finally:
        if orig is not None:
            os.environ["XDG_CONFIG_HOME"] = orig
        elif "XDG_CONFIG_HOME" in os.environ:
            del os.environ["XDG_CONFIG_HOME"]


def test_load_user_products_missing_key(tmp_path, capsys):
    products_dir = tmp_path / "gaiahealpixcache" / "products"
    products_dir.mkdir(parents=True)
    with open(products_dir / "incomplete.json", "w") as f:
        json.dump({"name": "incomplete"}, f)

    orig = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
    try:
        with pytest.raises(TypeError):
            _load_user_products()
    finally:
        if orig is not None:
            os.environ["XDG_CONFIG_HOME"] = orig
        elif "XDG_CONFIG_HOME" in os.environ:
            del os.environ["XDG_CONFIG_HOME"]


# ---------------------------------------------------------------------------
# Unregister non-existent product
# ---------------------------------------------------------------------------


def test_unregister_nonexistent():
    with pytest.raises(KeyError, match="not found"):
        unregister_product("does_not_exist_at_all")


# ---------------------------------------------------------------------------
# Cleanup cache early return
# ---------------------------------------------------------------------------


def test_cleanup_cache_no_dir(monkeypatch):
    def mock_get_cache_dir():
        return None

    monkeypatch.setattr("gaiahealpixcache.cache.get_cache_dir", mock_get_cache_dir)
    prod = GaiaProduct(
        name="orphan",
        url="http://example.com/",
        md5sum_file="_MD5SUM.txt",
        file_prefix="Test_",
        file_ext=".csv.gz",
        columns=["ra"],
    )
    _cleanup_cache(prod)
