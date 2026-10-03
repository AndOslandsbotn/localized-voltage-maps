# Localized Voltage Maps

Implementation of the algorithm described in `Structure_from_Voltage.pdf`
(included in this repo). A Python proof of concept lives in `src/lvm/`; a
C++ port is being built alongside it in `cpp/`.

## System requirements

Python package requirements are tracked in `requirements.txt` /
`pyproject.toml` and installed normally via pip. The pieces below are
system-level (not pip-installable) and are only needed for specific parts
of the project:

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

One environment for the library, the GPU baselines, the benchmarks and the tests (Python 3.14, CUDA 13):

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
git config core.hooksPath .githooks      # once per clone: refuses commits of large files
```

Add `.[cholmod]` if you've installed `libsuitesparse-dev` and want the CHOLMOD backend.

## Reproducing the experiments

Datasets are never stored in git. MNIST is downloaded from OpenML on first use, cached in `data/` (ignored) and checked against fingerprints, so every reproduction uses exactly the same images; the other datasets are generated. To fetch them up front:

```
.venv/bin/python benchmarks/common/datasets.py --download
```

Embeddings (`*.npz`) are not in git either: the experiment scripts regenerate them. See `benchmarks/README.md` for the layout and the order to run things in.

## C++ setup

```
cd cpp
cmake --preset default
cmake --build build
```
