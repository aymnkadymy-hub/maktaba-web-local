# Companion-paper experiments (deterministic)

Every number in the three companion papers (P1/P2/P3) is produced by these scripts.
They are **deterministic** (fixed seeds, eval mode) — pointing them at the same corpus
reproduces the same numbers. Results are written to [`results/`](results/) as JSON.

## Configuration (paths are environment-driven — no hard-coded locations)
| Variable | Default | Meaning |
|---|---|---|
| `MAKTABA_ROOT` | `~/maktaba-web-local` | path to the system source (for `import`ing `glossary`, `dialect_processor`, etc.) |
| `MAKTABA_CORPUS` | `~/maktaba-web-local/bm25_cache/corpus.json` | the chunked corpus the IR experiments read |

```bash
export MAKTABA_ROOT=/path/to/maktaba-web-local
export MAKTABA_CORPUS=$MAKTABA_ROOT/bm25_cache/corpus.json
python exp_p1_starvation.py        # P1: tenant starvation curves
python exp_p1b_ablation.py         # P1: crowding vs statistics-capture decomposition (+ bootstrap CI)
python exp_p1c_dense.py            # P1: measured dense FAISS post-filter baseline
python exp_p2_glossary.py          # P2: cross-script recall + continual learning
python exp_p2b_ablation.py         # P2: glossary-size ablation
python exp_p2c_labse.py            # P2: dense baselines (MiniLM, LaBSE)
python exp_p3_dialect.py           # P3: corruption safety + coverage/stability
python exp_p3b_dialectness.py      # P3: ALDi dialectness (honest negative result)
python exp_perf.py                 # latency + memory for all three mechanisms
```

## Note on data
The evaluation corpus is **private, user-owned books and is not redistributed**.
These scripts reproduce the *methodology* on any comparable local library; because the
methods are label-free and deterministic, the same corpus yields the same numbers.
The figures in each paper's `figures/` were generated from the JSON in `results/`.
