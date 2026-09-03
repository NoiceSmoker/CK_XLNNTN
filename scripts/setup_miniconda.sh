#!/usr/bin/env bash
# Cài Miniconda trong WSL (idempotent).
set -e
cd "$HOME"
if [ -d "$HOME/miniconda3" ]; then
  echo "Miniconda already installed at $HOME/miniconda3"
else
  echo "Downloading Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  echo "Installing Miniconda to $HOME/miniconda3 ..."
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  rm -f /tmp/miniconda.sh
fi
"$HOME/miniconda3/bin/conda" init bash >/dev/null 2>&1 || true
"$HOME/miniconda3/bin/conda" --version
echo "DONE_MINICONDA"
