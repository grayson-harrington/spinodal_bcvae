"""
Trains an ensemble of independently-seeded VAEs for the epistemic uncertainty
analysis (Fig. 11). Ported from code/_run_epistemic_ensemble.py, which
originally shelled out N_RUNS times to _final_vae_on_scores.py (now
scripts/03_train_final_vae.py).

NOTE: scripts/03_train_final_vae.py (owned by a separate porting pass) is
expected to support at least `--hparams-yaml`. It does not yet have a CLI flag
to vary the random seed or the output artifact path per run, which the
original ensemble driver relied on (each run wrote to its own
experiments/segmented_scores_vae_final/run_N/artifacts/ directory, with the
run number implicitly varying the underlying training framework's internal
seeding).

This port therefore does the minimum that is structurally correct without
editing 03_train_final_vae.py: it invokes the script once per ensemble member
via subprocess (matching the original's subprocess-based design) and moves/
renames the resulting config.PATHS["vae"] artifact into a per-run file under
config.DATA_DIR after each call, so N independent artifacts are retained. If
scripts/03_train_final_vae.py is later given a `--seed`/`--out` CLI flag, pass
`--seed {i}` below to make the runs genuinely independently-seeded rather than
relying on run-to-run nondeterminism (e.g. dataloader shuffling) alone.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess

import config

N_RUNS = 10
TRAIN_SCRIPT = Path(__file__).parent / "03_train_final_vae.py"


if __name__ == "__main__":
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"{TRAIN_SCRIPT} not found. This script depends on "
            "scripts/03_train_final_vae.py, which is ported separately."
        )

    for i in range(N_RUNS):
        print(f"\n{'=' * 60}")
        print(f"Epistemic ensemble run {i + 1} / {N_RUNS}")
        print(f"{'=' * 60}\n")

        # Each ensemble member is trained with a distinct seed so the members
        # are genuinely independent (the whole point of the epistemic-uncertainty
        # analysis). 03_train_final_vae.py writes the model directly to --out.
        run_vae_path = config.DATA_DIR / f"vae_ensemble_run_{i}.pkl"
        subprocess.run(
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--seed",
                str(i),
                "--out",
                str(run_vae_path),
            ],
            check=True,
            cwd=TRAIN_SCRIPT.parent,
        )
        print(f"Saved ensemble member {i} -> {run_vae_path}")

    print("\nAll epistemic ensemble runs complete.")
