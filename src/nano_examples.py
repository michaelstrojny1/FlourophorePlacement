import json
from typing import Dict, Any, List, Tuple

import enhanced_run


def run_scenario(
    name: str,
    n_refs: int,
    M: int,
    source_centers: List[Tuple[float, float]],
    train_steps: int,
    design_iters: int,
    D_range=(0.7, 1.5),
    a_range=(0.5, 2.0),
    seed: int = 42,
) -> Dict[str, Any]:
    results = enhanced_run.run_enhanced_experiment(
        n_refs=n_refs,
        M=M,
        source_centers=source_centers,
        train_steps=train_steps,
        design_iters=design_iters,
        D_range=D_range,
        a_range=a_range,
        sample_seed=seed,
    )

    enhanced_run.create_clear_visualizations(results, save_prefix=name)
    enhanced_run.generate_additional_figures(results, save_prefix=name)
    summary = enhanced_run.analyze_optimality(results)
    
    with open(f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    manifest = {
        "name": name,
        "n_refs": n_refs,
        "M": M,
        "source_centers": source_centers,
        "training_losses": results.get("training_losses", []),
        "criterion_final": summary.get("criterion_final"),
        "criterion_grid": summary.get("criterion_grid"),
        "criterion_random_mean": summary.get("criterion_random_mean"),
        "criterion_random_std": summary.get("criterion_random_std"),
        "grad_norm_final": summary.get("grad_norm_final"),
    }
    with open(f"{name}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    return manifest


def main():
    scenarios = []

    # Scenario 1: Diagonal sources, standard ranges, 16 sensors
    s1 = run_scenario(
        name="nano_s1",
        n_refs=5,
        M=16,
        source_centers=[(0.2, 0.8), (0.8, 0.2)],
        train_steps=1200,
        design_iters=300,
        D_range=(0.7, 1.5),
        a_range=(0.5, 2.0),
        seed=42,
    )
    scenarios.append(s1)

    # Scenario 2: Off-diagonal sources, narrower parameter ranges, 12 sensors
    s2 = run_scenario(
        name="nano_s2",
        n_refs=5,
        M=12,
        source_centers=[(0.3, 0.6), (0.7, 0.4)],
        train_steps=1000,
        design_iters=300,
        D_range=(0.8, 1.3),
        a_range=(0.6, 1.5),
        seed=1337,
    )
    scenarios.append(s2)

    # Scenario 3: Symmetric diagonal sources, more sensors (20)
    s3 = run_scenario(
        name="nano_s3",
        n_refs=5,
        M=20,
        source_centers=[(0.25, 0.25), (0.75, 0.75)],
        train_steps=1000,
        design_iters=300,
        D_range=(0.7, 1.5),
        a_range=(0.5, 2.0),
        seed=7,
    )
    scenarios.append(s3)

    # Scenario 4: Edge-biased sources, fewer sensors (10), narrower ranges
    s4 = run_scenario(
        name="nano_s4",
        n_refs=5,
        M=10,
        source_centers=[(0.15, 0.85), (0.85, 0.15)],
        train_steps=900,
        design_iters=300,
        D_range=(0.9, 1.4),
        a_range=(0.7, 1.3),
        seed=23,
    )
    scenarios.append(s4)

    with open("nano_summary.json", "w") as f:
        json.dump({"scenarios": scenarios}, f, indent=2)


if __name__ == "__main__":
    main()
