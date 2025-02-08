import os
import pathlib
import sys

def fix_path():
    """Fix path handling on Windows for Flair"""
    if sys.platform.startswith('win'):
        # Replace PosixPath with WindowsPath for torch loading
        def _new_from_parts(cls, args):
            return pathlib.WindowsPath(*args)
        pathlib.PosixPath._from_parts = classmethod(_new_from_parts)

# Apply fix when module is imported
fix_path() 