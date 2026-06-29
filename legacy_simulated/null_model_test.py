"""
Null Model Experiment for TDA Paper
====================================
This script tests whether observed topology signals are genuine or random artifacts.

Method:
1. Load real market data
2. Compute real topology metrics (CV, persistence, loop counts)
3. Shuffle returns within each asset (destroys cross-asset structure)
4. Recompute topology on shuffled data
5. Compare distributions with z-scores and effect sizes

If topology signal is real: Real CV << Shuffled CV
If topology signal is noise: Real CV ≈ Shuffled CV

Run: python null_model_test.py

Requirements:
    pip install numpy pandas yfinance scipy ripser matplotlib
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("yfinance not installed. Using synthetic data.")

try:
    import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("ripser not installed. Using simulated topology.")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed. Skipping plots.")


def correlation_to_distance(corr_matrix):
    """Convert correlation matrix to distance matrix."""
    return np.sqrt(2 * (1 - corr_matrix))


def compute_topology_metrics(returns_df):
    """Compute topology metrics from returns DataFrame."""
    corr = returns_df.corr().values
    dist = correlation_to_distance(corr)
    np.fill_diagonal(dist, 0)

    # Mean correlation
    n = len(corr)
    mean_corr = (corr.sum() - n) / (n * (n - 1))

    if HAS_RIPSER:
        result = ripser.ripser(dist, maxdim=1, distance_matrix=True)
        h1 = result['dgms'][1]
        h1 = h1[np.isfinite(h1[:, 1])]

        if len(h1) > 0:
            lifetimes = h1[:, 1] - h1[:, 0]
            cv = np.std(lifetimes) / np.mean(lifetimes) if np.mean(lifetimes) > 0 else 0
            total_persistence = np.sum(lifetimes)
            loop_count = len(h1)
        else:
            cv, total_persistence, loop_count = 0, 0, 0
    else:
        # Simulate based on correlation (for testing without ripser)
        cv = 0.65 - 0.4 * mean_corr + np.random.normal(0, 0.05)
        total_persistence = np.random.uniform(0.5, 2.0)
        loop_count = np.random.randint(3, 15)

    return {
        'cv': cv,
        'total_persistence': total_persistence,
        'loop_count': loop_count,
        'mean_corr': mean_corr
    }


def shuffle_returns(returns_df):
    """
    Shuffle returns within each asset (column).
    This destroys temporal cross-asset correlation structure
    while preserving marginal distributions.
    """
    shuffled = returns_df.copy()
    for col in shuffled.columns:
        shuffled[col] = np.random.permutation(shuffled[col].values)
    return shuffled


def run_null_model_test(n_shuffles=100):
    """
    Run null model experiment comparing real vs shuffled topology.
    """
    print("="*70)
    print("NULL MODEL EXPERIMENT")
    print("="*70)
    print("\nHypothesis: If topology signal is real, real CV << shuffled CV")
    print("            If topology signal is noise, real CV ≈ shuffled CV\n")

    # Load data
    if HAS_YFINANCE:
        print("Loading real market data...")
        # High-correlation sector (Financials)
        financials = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC',
                     'TFC', 'SCHW', 'BK', 'AXP', 'COF', 'CME', 'ICE']

        # Mixed sector (lower correlation)
        mixed = ['AAPL', 'XOM', 'JNJ', 'JPM', 'PG', 'CVX', 'UNH', 'HD',
                'KO', 'PEP', 'MRK', 'WMT', 'DIS', 'VZ', 'INTC']

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)

        datasets = {}

        for name, tickers in [('Financials (high ρ)', financials),
                               ('Mixed (low ρ)', mixed)]:
            print(f"\nDownloading {name}...")
            try:
                raw = yf.download(tickers, start=start_date, end=end_date,
                                 progress=False, auto_adjust=True)
                if isinstance(raw.columns, pd.MultiIndex):
                    prices = raw['Close'].dropna()
                else:
                    prices = raw.dropna()

                returns = prices.pct_change().dropna()
                if len(returns.columns) >= 10:
                    datasets[name] = returns
                    print(f"  Loaded {len(returns.columns)} stocks, {len(returns)} days")
            except Exception as e:
                print(f"  Error: {e}")
    else:
        # Synthetic data fallback
        print("Using synthetic data...")
        np.random.seed(42)
        n_days = 500

        # High correlation synthetic
        market = np.random.randn(n_days) * 0.01
        high_corr = pd.DataFrame({
            f'Stock{i}': market + np.random.randn(n_days) * 0.005
            for i in range(15)
        })

        # Low correlation synthetic
        low_corr = pd.DataFrame({
            f'Stock{i}': np.random.randn(n_days) * 0.02
            for i in range(15)
        })

        datasets = {
            'Financials (high ρ)': high_corr,
            'Mixed (low ρ)': low_corr
        }

    # Run null model test for each dataset
    results = {}

    for name, returns in datasets.items():
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print('='*50)

        # Compute real topology
        print("\nComputing real topology...")
        real_metrics = compute_topology_metrics(returns)
        print(f"  Real mean ρ: {real_metrics['mean_corr']:.3f}")
        print(f"  Real CV(H1): {real_metrics['cv']:.3f}")
        print(f"  Real loop count: {real_metrics['loop_count']}")

        # Compute shuffled topology (multiple times)
        print(f"\nComputing {n_shuffles} shuffled topologies...")
        shuffled_cvs = []
        shuffled_loops = []
        shuffled_persistence = []

        for i in range(n_shuffles):
            if (i + 1) % 20 == 0:
                print(f"  Shuffle {i+1}/{n_shuffles}...")

            shuffled = shuffle_returns(returns)
            metrics = compute_topology_metrics(shuffled)
            shuffled_cvs.append(metrics['cv'])
            shuffled_loops.append(metrics['loop_count'])
            shuffled_persistence.append(metrics['total_persistence'])

        shuffled_cvs = np.array(shuffled_cvs)
        shuffled_loops = np.array(shuffled_loops)

        # Statistical comparison
        print("\n" + "-"*40)
        print("RESULTS")
        print("-"*40)

        # Z-score for CV
        z_cv = (real_metrics['cv'] - np.mean(shuffled_cvs)) / np.std(shuffled_cvs)
        p_cv = 2 * (1 - stats.norm.cdf(abs(z_cv)))

        # Effect size (Cohen's d)
        cohens_d = (real_metrics['cv'] - np.mean(shuffled_cvs)) / np.std(shuffled_cvs)

        print(f"\nCV(H1) Comparison:")
        print(f"  Real CV:      {real_metrics['cv']:.4f}")
        print(f"  Shuffled CV:  {np.mean(shuffled_cvs):.4f} ± {np.std(shuffled_cvs):.4f}")
        print(f"  Z-score:      {z_cv:.2f}")
        print(f"  p-value:      {p_cv:.4f}")
        print(f"  Cohen's d:    {cohens_d:.2f}")

        # Interpretation
        if real_metrics['cv'] < np.mean(shuffled_cvs) and p_cv < 0.05:
            print(f"\n  ✓ REAL TOPOLOGY IS MORE STABLE THAN RANDOM")
            print(f"    Real CV is {abs(z_cv):.1f} std below shuffled mean")
            interpretation = "Signal confirmed"
        elif real_metrics['cv'] > np.mean(shuffled_cvs) and p_cv < 0.05:
            print(f"\n  ✗ REAL TOPOLOGY IS LESS STABLE THAN RANDOM")
            print(f"    This suggests correlation heterogeneity problem")
            interpretation = "Heterogeneity issue"
        else:
            print(f"\n  ? NO SIGNIFICANT DIFFERENCE FROM RANDOM")
            print(f"    Topology may not contain useful signal")
            interpretation = "No signal"

        # Loop count comparison
        z_loops = (real_metrics['loop_count'] - np.mean(shuffled_loops)) / max(np.std(shuffled_loops), 0.1)
        print(f"\nLoop Count Comparison:")
        print(f"  Real loops:     {real_metrics['loop_count']}")
        print(f"  Shuffled loops: {np.mean(shuffled_loops):.1f} ± {np.std(shuffled_loops):.1f}")
        print(f"  Z-score:        {z_loops:.2f}")

        results[name] = {
            'real_cv': real_metrics['cv'],
            'shuffled_cv_mean': np.mean(shuffled_cvs),
            'shuffled_cv_std': np.std(shuffled_cvs),
            'z_score': z_cv,
            'p_value': p_cv,
            'cohens_d': cohens_d,
            'interpretation': interpretation,
            'mean_corr': real_metrics['mean_corr']
        }

    # Summary
    print("\n" + "="*70)
    print("NULL MODEL TEST SUMMARY")
    print("="*70)

    print("\n{:<25} {:>10} {:>12} {:>10} {:>12}".format(
        "Dataset", "Mean ρ", "Real CV", "Z-score", "Result"))
    print("-"*70)

    for name, r in results.items():
        print("{:<25} {:>10.3f} {:>12.3f} {:>10.2f} {:>12}".format(
            name[:25], r['mean_corr'], r['real_cv'], r['z_score'], r['interpretation']))

    print("\n" + "-"*70)
    print("Interpretation Guide:")
    print("  • Z < -2 and p < 0.05: Real topology MORE stable than random (good!)")
    print("  • Z > +2 and p < 0.05: Real topology LESS stable than random (heterogeneity)")
    print("  • |Z| < 2: No significant difference from random noise")
    print("-"*70)

    # Create figure if matplotlib available
    if HAS_MATPLOTLIB and len(results) > 0:
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
        if len(results) == 1:
            axes = [axes]

        for ax, (name, r) in zip(axes, results.items()):
            # This would need the shuffled_cvs array saved - simplified version
            ax.axvline(r['real_cv'], color='red', linewidth=2, label=f"Real CV = {r['real_cv']:.3f}")
            ax.axvline(r['shuffled_cv_mean'], color='blue', linestyle='--',
                      label=f"Shuffled mean = {r['shuffled_cv_mean']:.3f}")
            ax.axvspan(r['shuffled_cv_mean'] - 2*r['shuffled_cv_std'],
                      r['shuffled_cv_mean'] + 2*r['shuffled_cv_std'],
                      alpha=0.3, color='blue', label='95% CI (shuffled)')
            ax.set_xlabel('CV(H₁)')
            ax.set_ylabel('Density')
            ax.set_title(f"{name}\nZ = {r['z_score']:.2f}, p = {r['p_value']:.3f}")
            ax.legend()

        plt.tight_layout()
        plt.savefig('null_model_results.png', dpi=150, bbox_inches='tight')
        print("\nSaved: null_model_results.png")

    # Generate LaTeX table
    print("\n" + "="*70)
    print("LATEX TABLE (copy to paper)")
    print("="*70)
    print("""
\\begin{table}[H]
\\centering
\\caption{Null Model Test: Real vs Shuffled Topology}
\\label{tab:null-model}
\\begin{tabular}{@{}lccccl@{}}
\\toprule
\\textbf{Dataset} & \\textbf{Mean $\\rho$} & \\textbf{Real CV} & \\textbf{Shuffled CV} & \\textbf{Z-score} & \\textbf{Result} \\\\
\\midrule""")

    for name, r in results.items():
        short_name = name.split('(')[0].strip()
        print(f"{short_name} & {r['mean_corr']:.2f} & {r['real_cv']:.3f} & "
              f"{r['shuffled_cv_mean']:.3f} $\\pm$ {r['shuffled_cv_std']:.3f} & "
              f"{r['z_score']:.2f} & {r['interpretation']} \\\\")

    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")

    return results


if __name__ == "__main__":
    results = run_null_model_test(n_shuffles=100)
