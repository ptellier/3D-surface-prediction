# 3D-surface-prediction

### Python Environment Setup

To create a new `virtual-env` environment called 'venv', install this repository's required packages in it, and activate it (on Mac).

```shell
python -m venv venv --python=python3.11
source ./venv/bin/activate
pip install .  # installs dependencies from pyproject.toml
```

Install torch separately specific to your system.
```shell
# on Windows with CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Download proprietary packages from Nexera Robotics after obtaining them from the wheel files.
You need the following packages:
 - `nexera_packages`
 - `neura_modules`

Do these installations without installing dependencies (using the `--no-deps` flag) since this causes
problems and all dependencies should already included in this repo's `pyproject.toml`. 

```shell
# install a package from a wheel (with no dependencies installed)
pip install ./path_to_wheel_folder/some_package.whl --no-deps
```