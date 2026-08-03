import drjit as dr
import pytest

def get_pkg(t):
    with dr.detail.scoped_rtld_deepbind():
        m = pytest.importorskip("local_ext")
    return pytest.get_backend_submodule(m, t)


@pytest.test_arrays('float32,shape=(*)')
def test01_lookup(t):
    pkg = get_pkg(t)

    assert dr.all(pkg.lookup(4, 5.0, 2) == 3.0)
    assert dr.all(pkg.lookup(4, 5.0, 4) == 5.0)
