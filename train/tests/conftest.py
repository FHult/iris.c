"""
Pytest configuration for train/ tests.

Adds the train/ directory to sys.path so that `from ip_adapter.xxx import yyy`
works without installing the package.

Also registers the test markers used to carve out the fast, dependency-free CI
subset (see `make test-ci`). Apply them with `@pytest.mark.<name>` on a test or
`pytestmark = pytest.mark.<name>` at module scope:

  slow            — long-running tests (mini training loops, large fixtures).
  requires_shards — needs real shard / dataset files on disk (not tmp fixtures).
  requires_mps    — needs the MLX/Metal GPU runtime (Apple Silicon only).
  quality         — runs golden-set / quality evaluation (model + GPU).
"""
import os
import sys

# train/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


_MARKERS = {
    "slow": "long-running test (e.g. mini training loop, large fixture build)",
    "requires_shards": "needs real shard/dataset files on disk (not tmp fixtures)",
    "requires_mps": "needs the MLX/Metal GPU runtime (Apple Silicon only)",
    "quality": "runs golden-set / quality evaluation (requires a model + GPU)",
}


def pytest_configure(config):
    for name, description in _MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")
