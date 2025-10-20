"""Setuptools integration for f90wrap.

This module provides a simple setuptools build_ext command that automatically
wraps and builds Fortran sources using f90wrap with Direct-C mode.

Usage in pyproject.toml:
    [build-system]
    requires = ["setuptools", "wheel", "numpy", "f90wrap"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "mypackage"

    [tool.f90wrap]
    sources = ["src/module1.f90", "src/module2.f90"]
    module_name = "mypackage"

Usage in setup.py:
    from setuptools import setup
    from f90wrap.setuptools_ext import F90WrapExtension, build_ext_cmdclass

    setup(
        name="mypackage",
        ext_modules=[
            F90WrapExtension(
                name="mymodule",
                sources=["src/module1.f90", "src/module2.f90"]
            )
        ],
        cmdclass=build_ext_cmdclass()
    )

    The package 'mypackage' will be auto-created and you can use: import mypackage
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext


class F90WrapExtension(Extension):
    """Extension class for f90wrap Fortran sources.

    Args:
        name: Name of the Python module (what you use in 'import name').
              Independent of Fortran module names in source files.
        sources: List of Fortran source files.
        kind_map: Optional path to kind_map file.
        package: Use package mode (-P flag).
        **kwargs: Additional Extension arguments.
    """

    def __init__(
        self,
        name: str,
        sources: List[str],
        kind_map: Optional[str] = None,
        package: bool = False,
        **kwargs
    ):
        self.f90wrap_sources = sources
        self.kind_map = kind_map
        self.package_mode = package
        super().__init__(name, sources=[], **kwargs)


class build_f90wrap_ext(_build_ext):
    """Custom build_ext command for F90WrapExtension."""

    def run(self):
        """Build f90wrap extensions."""
        f90wrap_exts = []
        other_exts = []

        for ext in self.extensions:
            if isinstance(ext, F90WrapExtension):
                f90wrap_exts.append(ext)
            else:
                other_exts.append(ext)

        for ext in f90wrap_exts:
            self.build_f90wrap(ext)

        self.extensions = other_exts
        super().run()

    def build_f90wrap(self, ext: F90WrapExtension):
        """Build a single f90wrap extension.

        Args:
            ext: F90WrapExtension to build.
        """
        from f90wrap import build as f90build

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        original_dir = Path.cwd()
        os.chdir(build_temp)

        try:
            cmd = ["f90wrap", "--direct-c", "-m", ext.name]

            if ext.kind_map:
                cmd.extend(["-k", str(Path(original_dir) / ext.kind_map)])

            if ext.package_mode:
                cmd.append("-P")

            abs_sources = [str(Path(original_dir) / src) for src in ext.f90wrap_sources]
            cmd.extend(abs_sources)

            self.announce(f"Running f90wrap: {' '.join(cmd)}", level=2)
            subprocess.run(cmd, check=True)

            self.announce(f"Building extension with Direct-C mode", level=2)
            ret = f90build.build_extension(
                module_name=ext.name,
                source_files=abs_sources,
                package_mode=ext.package_mode,
                verbose=self.verbose > 0
            )

            if ret != 0:
                raise RuntimeError(f"f90wrap build failed for {ext.name}")

            from distutils.sysconfig import get_config_var
            ext_suffix = get_config_var('EXT_SUFFIX') or '.so'

            package_name = self.distribution.metadata.name
            if self.inplace:
                pkg_root = Path(original_dir) / package_name
            else:
                build_lib_abs = Path(original_dir) / self.build_lib
                pkg_root = build_lib_abs / package_name

            pkg_root.mkdir(parents=True, exist_ok=True)

            c_ext_file = Path(f"_{ext.name}.so")
            if c_ext_file.exists():
                target_name = f"_{ext.name}{ext_suffix}"
                dest = pkg_root / target_name
                self.copy_file(str(c_ext_file), str(dest))

            py_file = Path(f"{ext.name}.py")
            if py_file.exists():
                with open(py_file, 'r') as f:
                    py_content = f.read()

                py_content = py_content.replace(
                    f'import _{ext.name}',
                    f'from . import _{ext.name}'
                )

                dest = pkg_root / py_file.name
                with open(dest, 'w') as f:
                    f.write(py_content)

            init_file = pkg_root / "__init__.py"
            if not init_file.exists():
                init_content = f"""\"\"\"Auto-generated package for f90wrap extension.\"\"\"\nfrom .{ext.name} import *\n"""
                with open(init_file, 'w') as f:
                    f.write(init_content)

            if ext.package_mode:
                pkg_dir = Path(ext.name)
                if pkg_dir.exists():
                    import shutil
                    dest_pkg = pkg_root / ext.name
                    if dest_pkg.exists():
                        shutil.rmtree(dest_pkg)
                    shutil.copytree(pkg_dir, dest_pkg)

        finally:
            os.chdir(original_dir)


def build_ext_cmdclass():
    """Return the custom build_ext command class.

    Returns:
        Dictionary suitable for setup(cmdclass=...).
    """
    return {"build_ext": build_f90wrap_ext}
