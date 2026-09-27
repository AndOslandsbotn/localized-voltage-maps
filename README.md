# Localized Voltage Maps

Implementation of the algorithm described in `Structure_from_Voltage.pdf`
(included in this repo). A Python proof of concept lives in `src/lvm/`; a
C++ port is being built alongside it in `cpp/`.

## System requirements

Python package requirements are tracked in `requirements.txt` /
`pyproject.toml` and installed normally via pip. The pieces below are
system-level (not pip-installable) and are only needed for specific parts
of the project:

- **`libsuitesparse-dev`** — needed to build the `scikit-sparse` Python
  package, which provides the CHOLMOD sparse Cholesky backend in
  `lvm.voltage` (`method="cholmod"`). Without it, that one backend raises
  an actionable `ImportError` at call time; everything else in `lvm`
  works fine without it.
  ```
  sudo apt-get install -y libsuitesparse-dev
  pip install scikit-sparse   # or: pip install -e ".[cholmod]"
  ```

- **`gfortran`** — needed to build LAPACK from source, which is a
  mandatory (non-optional) dependency of the `dlib` vcpkg port used by
  the `cpp/` build.
  ```
  sudo apt-get install -y gfortran
  ```

- **[vcpkg](https://github.com/microsoft/vcpkg)** (manifest mode) — manages
  the C++ dependencies (`eigen3`, `dlib`) declared in `cpp/vcpkg.json`.
  There's no apt package for it; it's bootstrapped from source:
  ```
  git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
  ~/vcpkg/bootstrap-vcpkg.sh
  export VCPKG_ROOT="$HOME/vcpkg"   # add to ~/.bashrc to persist
  ```
  `cpp/CMakePresets.json` reads `VCPKG_ROOT` to locate vcpkg's toolchain
  file, so it must be set before running `cmake --preset default`.

## Python setup

```
python3 -m venv .venv
.venv/bin/pip install -e ".[viz]"
```

Add `.[cholmod]` instead of (or alongside) `.[viz]` if you've installed
`libsuitesparse-dev` and want the CHOLMOD backend available too.

## C++ setup

```
cd cpp
cmake --preset default
cmake --build build
```
