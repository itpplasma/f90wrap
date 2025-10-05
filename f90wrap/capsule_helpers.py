"""
Utility to copy capsule_helpers.h to the build directory.
"""

import os
import shutil
from pathlib import Path


def copy_capsule_helpers(output_dir: str = ".") -> None:
    """
    Copy the capsule_helpers.h file to the specified output directory.

    Parameters
    ----------
    output_dir : str
        Directory where the header file should be copied
    """
    # Find the header file in the f90wrap package
    module_dir = Path(__file__).parent
    header_src = module_dir / "capsule_helpers.h"

    if not header_src.exists():
        raise FileNotFoundError(f"capsule_helpers.h not found in {module_dir}")

    # Copy to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    header_dst = output_path / "capsule_helpers.h"
    shutil.copy2(header_src, header_dst)

    print(f"Copied capsule_helpers.h to {header_dst}")


def get_capsule_helpers_path() -> Path:
    """
    Get the path to the capsule_helpers.h file.

    Returns
    -------
    Path
        Path to the capsule_helpers.h file
    """
    module_dir = Path(__file__).parent
    return module_dir / "capsule_helpers.h"