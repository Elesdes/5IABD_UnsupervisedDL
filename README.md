# 5AIBD Unsupervised Deep Learning Project

Syllabus is accessible [here](/docs/syllabus.pdf)

## Project Setup

### Install WSL Distribution

If you haven't installed WSL already, in a Windows terminal, run the following commands to install it and set it up:

```pwsh
wsl --install -d Ubuntu
wsl -s Ubuntu
```

#### From here, all commands are run in WSL

### Install Anaconda

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

After running the above commands, reboot WSL.

### Prepare Conda Environment

Before this step, you need to access the project's root directory with WSL.

```bash
conda update --all -y
conda create -n 5IABD_UnsupervisedDL python=3.11 -y
conda activate 5IABD_UnsupervisedDL
yes | pip install -r requirements.txt
```

### Setup Pre-Commit (Optional)

If you want to develop on this project, you must install pre-commit hooks. Else, it's very optional.

```bash
pre-commit install
```

### Setup File Tree

```bash
python src/setup.py
```
