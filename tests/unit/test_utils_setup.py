# tests/test_utils_setup.py
import os
import tempfile
from damping_wings import setup_output_dirs
from damping_wings.config import constants

def test_setup_output_dirs_creates_directories():
    """setup_output_dirs should create all required subdirectories"""
    with tempfile.TemporaryDirectory() as tmpdir:
        constants.newpath = tmpdir
        constants.plotpath = os.path.join(tmpdir, 'plots')
        constants.txt_files = os.path.join(tmpdir, 'txt_files')
        constants.cache_path = os.path.join(tmpdir, 'cache_files')
        setup_output_dirs()
        assert os.path.exists(constants.plotpath)
        assert os.path.exists(constants.txt_files)
        assert os.path.exists(constants.cache_path)