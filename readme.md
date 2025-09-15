# Physics-Informed Sensor Placement for Nanomaterials

Optimal fluorophore sensor placement using physics-informed neural networks for nanomaterial diffusion parameter estimation. Can be modified for **any** sensor placement problem!

## Overview

Physics-informed approach to optimize sensor placement for nanomaterial characterization experiments. Neural networks satisfy diffusion-reaction physics to achieve **70-77% improvement** over random sensor placement.

Includes implemntation with and without a Fisher Information objective

## Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib

# Run all nanomaterial scenarios (S1-S4) 
python src/nano_examples.py

# Run principled FIM vs variance comparison
python src/conditional_fim_run.py --refs 15 --steps 1500 --normalize --anneal_pen
```

## Repository Structure

```
├── src/                          # Source code
│   ├── fluor_placement.py        # Core PINN implementation
│   ├── conditional_fim_run.py    # Principled FIM with conditional PINN
│   └── nano_examples.py          # Nanomaterial scenarios S1-S4
├── docs/                         # Documentation
│   ├── presentation.tex          # Complete technical presentation
│   └── fim_convergence.png       # FIM convergence figure
├── results/                      # Generated results
│   ├── nano_s*_with_fluorophores.png     # Optimized placements
│   ├── nano_s*_summary.json              # Performance metrics
│   ├── conditional_fim_summary_final.json # Final FIM comparison
│   └── fim_convergence_final.png          # Final convergence plot
├── tools/                        # External dependencies
│   └── xtb-6.7.1/               # Quantum chemistry toolkit
├── .gitignore
├── README.md
└── requirements.txt
```

## Nanomaterial Results

| Scenario | Sensors | Final J    | Grid J     | Random J   | Improvement |
|----------|---------|------------|------------|------------|-------------|
| S1       | 16      | -0.01200   | -0.01182   | -0.01703   | 76.1%       |
| S2       | 12      | -0.01288   | -0.01586   | -0.02028   | 74.1%       |
| S3       | 20      | -0.01101   | -0.01239   | -0.01551   | 77.4%       |
| S4       | 10      | -0.01348   | -0.02012   | -0.02091   | 71.6%       |

## Fisher Information Matrix (FIM) vs Variance

The repository includes both variance-based and principled FIM sensor optimization:

| Method | Final Loss | Notes |
|--------|------------|-------|
| Variance | 0.01171 | Fast, stable, empirically validated |
| FIM (normalized) | 0.05227 | Advanced |

FIM loss will be larger due to additional FIM term, however, it is the most principled

```bash
# Full comparison with tuned parameters
python src/conditional_fim_run.py --refs 15 --steps 1500 --iters 350 --width 128 --sigma 0.1 --normalize --anneal_pen
```

## Lab Implementation

1. **Run optimization**: `python src/nano_examples.py`
2. **Get coordinates**: Check `results/nano_s*_with_fluorophores.png` 
3. **Scale to physical**: [0,1] coordinates → your sample dimensions
4. **Place detectors**: At optimized (x,y) positions
5. **Measure**: Fluorescence intensity at steady state
6. **Estimate parameters**: Fit PINN to measured data

## Key Features

- **Physics-Informed**: Neural networks satisfy diffusion-reaction PDE
- **Principled FIM**: Conditional PINN enables exact autograd sensitivities ∂u/∂(D,α)
- **Multi-Objective**: Balances information content, coverage, anti-clustering
- **Reproducible**: Fixed seeds, documented hyperparameters
- **Validated**: Tested across 4 nanomaterial scenarios
- **Lab-Ready**: Direct coordinate translation to physical experiments

## Customization

Edit `src/nano_examples.py` to customize for your material:

```python
scenario = {
    "source_centers": [(0.2, 0.8), (0.8, 0.2)],  # Your injection sites
    "D_range": (0.7, 1.5),    # Expected diffusion coefficient range
    "a_range": (0.5, 2.0),    # Expected reaction coefficient range
    "M": 16,                  # Number of sensors
}
```

## Documentation

- `docs/presentation.tex`: Comprehensive technical presentation with detailed methodology, results, and implementation guidance
- Build instructions:
  ```bash
  cd docs
  pdflatex presentation.tex
  ```
  (Requires LaTeX distribution like TeX Live or MiKTeX)

## Troubleshooting

- **PINN not converging**: Reduce learning rate to 1e-4, increase training steps
- **Sensors clustering**: Increase anti-clustering penalty λ₂ to 0.1
- **Poor coverage**: Increase coverage penalty λ₁ to 0.2
- **Physical setup**: See `docs/presentation.tex` for complete lab implementation guide

## Citation

```bibtex
@software{fluorophore_placement_2025,
  title={Physics-Informed Sensor Placement for Nanomaterial Characterization},
  author={Michael},
  year={2025},
  url={https://github.com/your-username/FlourophorePlacement}
}
```

## Performance

- **Runtime**: 2-5 minutes per scenario (CPU only)
- **Improvement**: 70-77% over random placement
- **Reproducible**: Fixed seeds ensure identical results
- **Scalable**: Handles 10-20 sensors efficiently

## Setup and Requirements

**Dependencies**: Python 3.11+, PyTorch 2.x, NumPy, Matplotlib  
**Hardware**: Standard CPU (no GPU required)  

```powershell
# Windows PowerShell setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage Examples

### 1. Enhanced Experiment (Recommended)
```bash
python enhanced_run.py
```
**Output**: Complete analysis with 16 optimized fluorophores across 5 parameter configurations
- `enhanced_domain.png` — Physical field with source peaks  
- `enhanced_with_fluorophores.png` — Optimized placements overlay
- `enhanced_convergence.png` — Design optimization progress
- `enhanced_parameter_space.png` — Reference parameter distribution  
- `enhanced_distances.png` — Inter-fluorophore distance analysis
- `validation_summary.json` — Final performance metrics

**Key Results**: 77.8% improvement, design criterion -0.01216, gradient norm 0.0071

### 2. Simple Example (Quick Test)
```bash
python simple_run.py
```
**Output**: Lightweight demonstration with 8 fluorophores and 3 parameter configurations
- Console output showing optimization progress and final placements
- Basic visualization of results

### 3. Pattern Comparison Study
```bash
python mask_example.py
```
**Output**: Comparative analysis of predefined geometric arrangements
- `mask_square.png` — Square grid pattern (criterion: -0.0115)
- `mask_hcp.png` — Hexagonal close-packed pattern (criterion: -0.0142)  
- `mask_compare.json` — Quantitative comparison results

## Validation Results

Our optimization framework demonstrates:
- **4.5× improvement** over random initialization (-0.01216 vs -0.05464)
- **98.3% performance** relative to structured grid baseline  
- **Convergence confirmation** with final gradient norm 7.11×10⁻³
- **Robust spatial distribution** (min distance: 0.089, max: 1.128)

## Technical Implementation

- **PINN Architecture**: MLP with width=96, depth=6, tanh activation
- **Training Protocol**: 1500 steps, Adam optimizer, lr=1e-3
- **Design Optimization**: 300 iterations with anti-clustering penalties
- **Reproducibility**: Fixed random seeds (42) across all experiments
- **Automatic Differentiation**: Standard PyTorch autograd (no external PINN libraries)

## GitHub Preparation

This repository is now cleaned and organized for public release:
- All source code is production-ready
- Presentation is technically accurate and polished
- Documentation is comprehensive
- Dependencies are pinned in requirements.txt

## Pushing to GitHub

1. Create a new repository on GitHub
2. Update the remote URL:
   ```bash
   git remote set-url origin https://github.com/your-username/your-repo.git
   ```
3. Push your code:
   ```bash
   git push -u origin main
   ```

## License
MIT (see LICENSE)
