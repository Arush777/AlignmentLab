# Prajna cluster GPU inventory

Queried from `login2.prajna.iitb.ac.in` via Slurm (`sinfo` / `scontrol` / `sbatch` + `nvidia-smi`) on **2026-07-22**.

Login nodes have **no GPUs**. Use `sbatch` / `srun` (interactive only on `-p interactive`).

## Totals

| | Count |
|---|---|
| Unique compute GPUs | **204** |
| DGX (A100) | 72 (9×8) |
| A40 | 76 (19×4) |
| L40S | 56 (7×8) |

Free counts change continuously; re-check with Slurm.

## Partitions & hardware (probed)

| Partition | GPUs (nodes) | GPU model | VRAM | CPUs/node | RAM/node | Max wall |
|-----------|--------------|-----------|------|-----------|----------|----------|
| `dgx` | 72 (9×8) | NVIDIA A100-SXM4-80GB | 80 GB (81920 MiB) | 256 | ~2 TB | 6 days |
| `dgx-mpi` | 24 of those (cn11–cn13) | A100-SXM4-80GB | 80 GB | 256 | ~2 TB | 6 days |
| `a40` | 76 (19×4) | NVIDIA A40 | ~45 GB (46068 MiB) | 64 | ~503 GB | 4 days |
| `l40` *(default)* | 56 (7×8) | NVIDIA L40S | ~45 GB (46068 MiB) | 32–64 | ~440–503 GB | 2 days |
| `interactive` | shared (cn11-dgx, cn22-a40, cn41-l40) | mix | mix | mix | mix | 4 hours |
| `debug` | 4 (cn39-a40) | A40 | ~45 GB | 64 | ~503 GB | 30 min |

Nodes: `cn11–cn19-dgx`, `cn21–cn39-a40`, `cn40–cn46-l40`.

Probe notes: A100 on `cn11-dgx` (driver 550.144.03, CC 8.0); A40 on `cn31-a40` and L40S on `cn46-l40` (driver 570.86.15, CC 8.6 / 8.9).

QoS names `l4` and `gh200` exist in accounting but had **no live partitions** at query time.

## Per-user QoS caps (typical)

| QoS | Max GPUs / job | Max running jobs | Max submit |
|-----|----------------|------------------|------------|
| `a40` | 2 | 3 | 6 |
| `l40` | 4 | 4 | 5 |
| `dgx` | 4 | 4 | 5 |
| `dgx-mpi` | 8 | 1 | 2 |
| `interactive` | 8 | 2 | 2 |

## Useful commands

```bash
sinfo -N -o "%N %P %G %T"
squeue -u $USER
scontrol show node cn46-l40
scontrol show partition
```

Batch example:

```bash
#SBATCH -p l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
```

Interactive:

```bash
srun -p interactive --qos=interactive --gres=gpu:1 -t 00:30:00 --pty bash
```
