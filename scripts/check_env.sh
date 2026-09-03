#!/usr/bin/env bash
echo '=== GPU in WSL ==='
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 | head -3
echo '=== nvcc ==='
nvcc --version 2>&1 | tail -2
echo '=== python3 ==='
python3 --version 2>&1
echo '=== conda ==='
(conda --version 2>&1 || echo 'no conda')
echo '=== miniconda/anaconda dir ==='
ls -d ~/miniconda3 ~/anaconda3 2>/dev/null || echo 'none'
echo '=== cpu/mem ==='
nproc
free -h | head -2
echo '=== distro ==='
. /etc/os-release && echo "$PRETTY_NAME"
