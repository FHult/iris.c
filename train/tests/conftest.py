"""
Pytest configuration for train/ tests.

Adds the train/ directory to sys.path so that `from ip_adapter.xxx import yyy`
works without installing the package.
"""
import os
import sys

# train/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
