# Jarvis Lab VM Deployment

This repository does not contain Jarvis Lab-specific provisioning code, but it now includes a minimal Linux VM workflow for syncing the repo to a remote GPU instance and preparing the Python environment there.

## Assumptions

- The Jarvis Lab instance already exists.
- You can reach it over SSH.
- The VM is Linux-based.
- NVIDIA drivers and a CUDA-capable PyTorch-compatible environment are already available at the system level.
- Required dataset files are either already present on the VM or will be transferred separately.

## Files added for VM workflow

- `scripts/deploy_linux_vm.sh` — sync the repository to a remote host with `rsync`, excluding local-only state.
- `scripts/bootstrap_linux_vm.sh` — create `.venv`, install requirements, and print Torch/CUDA visibility.

## Initial sync and bootstrap

Run this from the repository root on your local machine:

```bash
bash scripts/deploy_linux_vm.sh <ssh-target> <remote-dir>
```

Example:

```bash
bash scripts/deploy_linux_vm.sh user@gpu-vm ~/FakeNews
```

## Remote training flow

After bootstrap completes, SSH into the VM and run:

```bash
cd ~/FakeNews
source .venv/bin/activate
python src/train_hf_adapter.py \
  --dataset master_dataset \
  --model-id Qwen/Qwen3.6-35B-A3B \
  --output-dir artifacts/training/qwen_35b \
  --generate-model-folder \
  --model-folder artifacts/FakeNews_Model \
  --hf-repo-id JayNightmare/FakeNews
```

## Notes

- The deploy script intentionally excludes `artifacts/` because local training outputs are large and should not be copied by default.
- The deploy script also excludes `.venv/` so the VM builds its own environment.
- If the VM does not already contain the required datasets under `src/datasets/data/`, transfer them separately or regenerate them on the VM.
- This workflow prepares the repo for remote execution, but it does not provision GPU drivers, CUDA runtime, or Hugging Face authentication.
