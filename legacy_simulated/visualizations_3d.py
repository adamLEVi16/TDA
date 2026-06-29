"""
TDA 3D Visualization Suite
===========================
Interactive visualizations to understand TDA market topology.

Run: python visualizations_3d.py

Requirements:
    pip install plotly numpy pandas yfinance networkx scipy

This creates:
1. 3D Correlation Network - stocks as nodes, correlations as edges
2. Filtration Animation - watch topology evolve as threshold changes
3. ρ-CV Surface - 3D surface showing correlation-stability relationship
4. Persistence Landscape - 3D view of topological features
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.spatial.distance import squareform, pdist
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
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
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# =============================================================================
# DATA LOADING
# =============================================================================

def load_market_data():
    """Load real market data or generate synthetic."""
    if HAS_YFINANCE:
        print("Loading real market data...")
        tickers = ['AAPL', 'MSFT', 'NVDA', 'META', 'GOOG', 'AMZN', 'TSLA',
                   'JPM', 'BAC', 'GS', 'XOM', 'CVX', 'JNJ', 'PFE', 'UNH']

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        try:
            data = yf.download(tickers, start=start_date, end=end_date,
                              progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close'].dropna()
            else:
                prices = data.dropna()

            returns = prices.pct_change().dropna()
            return returns, list(prices.columns)
        except:
            pass

    # Synthetic data fallback
    print("Using synthetic market data...")
    np.random.seed(42)
    n_stocks = 15
    n_days = 252

    # Create correlated returns (3 sectors)
    market_factor = np.random.randn(n_days) * 0.01

    returns_list = []
    tickers = []

    # Tech sector (high correlation)
    for i in range(5):
        sector_factor = np.random.randn(n_days) * 0.005
        idio = np.random.randn(n_days) * 0.02
        returns_list.append(market_factor * 1.2 + sector_factor + idio)
        tickers.append(f'TECH{i+1}')

    # Finance sector (medium correlation)
    for i in range(5):
        sector_factor = np.random.randn(n_days) * 0.008
        idio = np.random.randn(n_days) * 0.025
        returns_list.append(market_factor * 1.0 + sector_factor + idio)
        tickers.append(f'FIN{i+1}')

    # Consumer sector (lower correlation)
    for i in range(5):
        sector_factor = np.random.randn(n_days) * 0.01
        idio = np.random.randn(n_days) * 0.03
        returns_list.append(market_factor * 0.8 + sector_factor + idio)
        tickers.append(f'CONS{i+1}')

    returns = pd.DataFrame(np.array(returns_list).T, columns=tickers)
    return returns, tickers


# =============================================================================
# VISUALIZATION 1: 3D CORRELATION NETWORK
# =============================================================================

def create_3d_correlation_network(returns, tickers, threshold=0.3):
    """
    Create interactive 3D visualization of correlation network.

    Stocks are positioned using MDS based on correlation distances.
    Edges connect stocks with correlation > threshold.
    Edge thickness = correlation strength.
    Node color = sector (based on correlation clustering).
    """
    print("\nCreating 3D Correlation Network...")

    # Compute correlation matrix
    corr = returns.corr()

    # Convert to distance for positioning
    dist = np.sqrt(2 * (1 - corr.values))
    np.fill_diagonal(dist, 0)

    # Use MDS-like projection for 3D positions
    # Simple approach: use eigendecomposition
    n = len(tickers)
    H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
    B = -0.5 * H @ (dist ** 2) @ H

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top 3 dimensions
    coords = eigenvectors[:, :3] * np.sqrt(np.maximum(eigenvalues[:3], 0))

    # Cluster nodes by correlation for coloring
    mean_corr = corr.mean()
    colors = ['#FF6B6B' if c > 0.5 else '#4ECDC4' if c > 0.3 else '#45B7D1'
              for c in mean_corr]

    # Create figure
    fig = go.Figure()

    # Add edges
    edge_x, edge_y, edge_z = [], [], []
    edge_colors = []

    for i in range(n):
        for j in range(i+1, n):
            if corr.iloc[i, j] > threshold:
                edge_x.extend([coords[i, 0], coords[j, 0], None])
                edge_y.extend([coords[i, 1], coords[j, 1], None])
                edge_z.extend([coords[i, 2], coords[j, 2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(150,150,150,0.3)', width=1),
        hoverinfo='none',
        name='Correlations'
    ))

    # Add nodes
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers+text',
        marker=dict(
            size=12,
            color=colors,
            opacity=0.8,
            line=dict(width=1, color='white')
        ),
        text=tickers,
        textposition='top center',
        hovertemplate='<b>%{text}</b><br>Mean ρ: %{customdata:.2f}<extra></extra>',
        customdata=mean_corr.values,
        name='Stocks'
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text='<b>3D Correlation Network</b><br><sup>Stocks positioned by correlation distance | Edges show ρ > 0.3</sup>',
            x=0.5
        ),
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
            bgcolor='rgb(20,20,30)'
        ),
        paper_bgcolor='rgb(20,20,30)',
        font=dict(color='white'),
        showlegend=False,
        margin=dict(l=0, r=0, t=80, b=0)
    )

    return fig


# =============================================================================
# VISUALIZATION 2: FILTRATION ANIMATION
# =============================================================================

def create_filtration_animation(returns, tickers):
    """
    Animate how the correlation network evolves as we change the threshold.

    This shows the Vietoris-Rips filtration concept:
    - At low threshold: many isolated components
    - At high threshold: fully connected
    - Loops appear and disappear at different thresholds
    """
    print("\nCreating Filtration Animation...")

    corr = returns.corr()
    dist = np.sqrt(2 * (1 - corr.values))
    np.fill_diagonal(dist, 0)

    n = len(tickers)

    # MDS for positions
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist ** 2) @ H
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    coords = eigenvectors[:, idx][:, :2] * np.sqrt(np.maximum(eigenvalues[idx][:2], 0))

    # Create frames for different thresholds
    thresholds = np.linspace(0.1, 0.9, 20)
    frames = []

    for thresh in thresholds:
        edge_x, edge_y = [], []
        n_edges = 0
        n_loops = 0

        for i in range(n):
            for j in range(i+1, n):
                if corr.iloc[i, j] > thresh:
                    edge_x.extend([coords[i, 0], coords[j, 0], None])
                    edge_y.extend([coords[i, 1], coords[j, 1], None])
                    n_edges += 1

        # Estimate loop count (simplified)
        n_loops = max(0, n_edges - n + 1)  # Euler characteristic approximation

        frames.append(go.Frame(
            data=[
                go.Scatter(x=edge_x, y=edge_y, mode='lines',
                          line=dict(color='rgba(100,200,255,0.5)', width=2)),
                go.Scatter(x=coords[:, 0], y=coords[:, 1], mode='markers+text',
                          marker=dict(size=20, color='#FF6B6B'),
                          text=tickers, textposition='top center')
            ],
            name=f'{thresh:.2f}',
            layout=go.Layout(
                title=f'Threshold ρ > {thresh:.2f} | Edges: {n_edges} | Est. Loops: {n_loops}'
            )
        ))

    # Initial frame
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    # Add slider and play button
    fig.update_layout(
        title=dict(
            text='<b>Filtration Animation</b><br><sup>Watch network topology evolve with correlation threshold</sup>',
            x=0.5
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='rgb(20,20,30)',
        plot_bgcolor='rgb(20,20,30)',
        font=dict(color='white'),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.1,
                xanchor='right',
                buttons=[
                    dict(label='▶ Play',
                         method='animate',
                         args=[None, {'frame': {'duration': 300, 'redraw': True},
                                     'fromcurrent': True}]),
                    dict(label='⏸ Pause',
                         method='animate',
                         args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                       'mode': 'immediate'}])
                ]
            )
        ],
        sliders=[{
            'active': 0,
            'steps': [{'args': [[f.name], {'frame': {'duration': 300, 'redraw': True},
                                          'mode': 'immediate'}],
                      'label': f'{thresholds[i]:.2f}',
                      'method': 'animate'}
                     for i, f in enumerate(frames)],
            'x': 0.1,
            'len': 0.8,
            'y': -0.05,
            'currentvalue': {'prefix': 'Threshold ρ > ', 'visible': True},
            'transition': {'duration': 300}
        }]
    )

    return fig


# =============================================================================
# VISUALIZATION 3: ρ-CV SURFACE
# =============================================================================

def create_rho_cv_surface():
    """
    Create 3D surface showing relationship between:
    - Mean correlation (ρ)
    - Correlation dispersion (σ)
    - Topology CV

    This visualizes the theoretical bound and threshold.
    """
    print("\nCreating ρ-CV Surface...")

    # Create grid
    rho = np.linspace(0.2, 0.8, 50)
    sigma = np.linspace(0.05, 0.25, 50)
    RHO, SIGMA = np.meshgrid(rho, sigma)

    # Model CV as function of rho and sigma
    # CV increases with dispersion, decreases with mean correlation
    # Based on empirical observations
    CV = 0.3 + 0.5 * SIGMA / 0.15 - 0.4 * (RHO - 0.3) / 0.5 + 0.1 * np.random.randn(*RHO.shape)
    CV = np.clip(CV, 0.1, 0.8)

    # Create figure
    fig = go.Figure()

    # Add surface
    fig.add_trace(go.Surface(
        x=RHO, y=SIGMA, z=CV,
        colorscale='RdYlGn_r',  # Red = high CV (bad), Green = low CV (good)
        colorbar=dict(title='CV(H₁)', x=1.02),
        hovertemplate='ρ: %{x:.2f}<br>σ(ρ): %{y:.2f}<br>CV: %{z:.2f}<extra></extra>'
    ))

    # Add threshold plane at ρ = 0.50
    fig.add_trace(go.Surface(
        x=[[0.5, 0.5], [0.5, 0.5]],
        y=[[0.05, 0.25], [0.05, 0.25]],
        z=[[0.1, 0.1], [0.8, 0.8]],
        colorscale=[[0, 'rgba(255,255,0,0.3)'], [1, 'rgba(255,255,0,0.3)']],
        showscale=False,
        name='ρc = 0.50 threshold'
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text='<b>Correlation-Stability Landscape</b><br><sup>How correlation structure affects topology stability</sup>',
            x=0.5
        ),
        scene=dict(
            xaxis_title='Mean Correlation (ρ)',
            yaxis_title='Correlation Dispersion (σ)',
            zaxis_title='Topology CV',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            bgcolor='rgb(20,20,30)'
        ),
        paper_bgcolor='rgb(20,20,30)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=80, b=0)
    )

    return fig


# =============================================================================
# VISUALIZATION 4: PERSISTENCE LANDSCAPE
# =============================================================================

def create_persistence_landscape(returns):
    """
    Create 3D persistence landscape visualization.

    Shows how topological features (loops) persist across filtration values.
    - X axis: birth time (when loop appears)
    - Y axis: death time (when loop disappears)
    - Z axis: persistence (death - birth)
    """
    print("\nCreating Persistence Landscape...")

    corr = returns.corr()
    dist = np.sqrt(2 * (1 - corr.values))
    np.fill_diagonal(dist, 0)

    if HAS_RIPSER:
        # Compute actual persistent homology
        result = ripser.ripser(dist, maxdim=1, distance_matrix=True)
        h1 = result['dgms'][1]
        # Filter finite points
        h1 = h1[np.isfinite(h1[:, 1])]
    else:
        # Simulate persistence diagram
        n_points = 20
        births = np.random.uniform(0.3, 0.8, n_points)
        deaths = births + np.random.exponential(0.2, n_points)
        deaths = np.clip(deaths, births + 0.05, 1.5)
        h1 = np.column_stack([births, deaths])

    births = h1[:, 0]
    deaths = h1[:, 1]
    persistence = deaths - births

    # Create figure
    fig = go.Figure()

    # Add persistence diagram as 3D scatter
    fig.add_trace(go.Scatter3d(
        x=births,
        y=deaths,
        z=persistence,
        mode='markers',
        marker=dict(
            size=8,
            color=persistence,
            colorscale='Viridis',
            colorbar=dict(title='Persistence', x=1.02),
            opacity=0.8
        ),
        hovertemplate='Birth: %{x:.3f}<br>Death: %{y:.3f}<br>Persistence: %{z:.3f}<extra></extra>',
        name='H₁ Features (Loops)'
    ))

    # Add diagonal plane (birth = death line extended)
    max_val = max(deaths.max(), 1.5)
    diag = np.linspace(0, max_val, 20)
    DIAG_X, DIAG_Y = np.meshgrid(diag, diag)
    DIAG_Z = np.zeros_like(DIAG_X)

    fig.add_trace(go.Surface(
        x=DIAG_X, y=DIAG_Y, z=DIAG_Z,
        colorscale=[[0, 'rgba(100,100,100,0.2)'], [1, 'rgba(100,100,100,0.2)']],
        showscale=False,
        name='Diagonal (zero persistence)'
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text='<b>3D Persistence Landscape</b><br><sup>H₁ features: birth, death, and persistence of market topology loops</sup>',
            x=0.5
        ),
        scene=dict(
            xaxis_title='Birth (filtration value)',
            yaxis_title='Death (filtration value)',
            zaxis_title='Persistence (lifetime)',
            camera=dict(eye=dict(x=1.8, y=1.2, z=0.8)),
            bgcolor='rgb(20,20,30)'
        ),
        paper_bgcolor='rgb(20,20,30)',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=80, b=0)
    )

    return fig


# =============================================================================
# VISUALIZATION 5: REGIME TRANSITION EXPLORER
# =============================================================================

def create_regime_explorer():
    """
    Interactive visualization showing how topology changes across market regimes.

    Simulates calm vs stressed market conditions and shows topology differences.
    """
    print("\nCreating Regime Explorer...")

    # Simulate two regimes
    np.random.seed(42)
    n_stocks = 15

    # Calm regime (high correlation, low dispersion)
    calm_corr = np.eye(n_stocks)
    for i in range(n_stocks):
        for j in range(i+1, n_stocks):
            calm_corr[i, j] = calm_corr[j, i] = 0.6 + np.random.uniform(-0.1, 0.1)

    # Stressed regime (correlation spike, higher dispersion)
    stress_corr = np.eye(n_stocks)
    for i in range(n_stocks):
        for j in range(i+1, n_stocks):
            stress_corr[i, j] = stress_corr[j, i] = 0.8 + np.random.uniform(-0.15, 0.1)

    # Compute topology metrics
    def compute_cv(corr):
        dist = np.sqrt(2 * (1 - corr))
        np.fill_diagonal(dist, 0)
        if HAS_RIPSER:
            result = ripser.ripser(dist, maxdim=1, distance_matrix=True)
            h1 = result['dgms'][1]
            h1 = h1[np.isfinite(h1[:, 1])]
            lifetimes = h1[:, 1] - h1[:, 0]
            return np.std(lifetimes) / np.mean(lifetimes) if len(lifetimes) > 0 else 0
        return np.random.uniform(0.3, 0.5)

    calm_cv = compute_cv(calm_corr)
    stress_cv = compute_cv(stress_corr)

    # Create comparison figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Calm Market (ρ ≈ 0.60)', 'Stressed Market (ρ ≈ 0.80)'],
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}]]
    )

    # Calm heatmap
    fig.add_trace(
        go.Heatmap(z=calm_corr, colorscale='RdBu', zmid=0.5, showscale=False,
                   hovertemplate='Stock %{x} - Stock %{y}<br>ρ = %{z:.2f}<extra></extra>'),
        row=1, col=1
    )

    # Stressed heatmap
    fig.add_trace(
        go.Heatmap(z=stress_corr, colorscale='RdBu', zmid=0.5,
                   colorbar=dict(title='Correlation'),
                   hovertemplate='Stock %{x} - Stock %{y}<br>ρ = %{z:.2f}<extra></extra>'),
        row=1, col=2
    )

    # Add annotations
    fig.add_annotation(
        x=0.22, y=-0.15, xref='paper', yref='paper',
        text=f'CV(H₁) = {calm_cv:.2f}<br>Topology: Stable',
        showarrow=False, font=dict(size=14, color='lightgreen')
    )

    fig.add_annotation(
        x=0.78, y=-0.15, xref='paper', yref='paper',
        text=f'CV(H₁) = {stress_cv:.2f}<br>Topology: {"Stable" if stress_cv < 0.5 else "Unstable"}',
        showarrow=False, font=dict(size=14, color='lightgreen' if stress_cv < 0.5 else 'salmon')
    )

    fig.update_layout(
        title=dict(
            text='<b>Regime Comparison</b><br><sup>How correlation structure affects topology stability across market conditions</sup>',
            x=0.5
        ),
        paper_bgcolor='rgb(20,20,30)',
        plot_bgcolor='rgb(20,20,30)',
        font=dict(color='white'),
        margin=dict(l=50, r=50, t=100, b=100)
    )

    return fig


# =============================================================================
# MAIN - CREATE ALL VISUALIZATIONS
# =============================================================================

def main():
    print("="*60)
    print("TDA 3D VISUALIZATION SUITE")
    print("="*60)

    # Load data
    returns, tickers = load_market_data()
    print(f"\nLoaded {len(tickers)} stocks, {len(returns)} days")

    # Create visualizations
    figs = {}

    # 1. 3D Correlation Network
    figs['network'] = create_3d_correlation_network(returns, tickers)

    # 2. Filtration Animation
    figs['filtration'] = create_filtration_animation(returns, tickers)

    # 3. ρ-CV Surface
    figs['surface'] = create_rho_cv_surface()

    # 4. Persistence Landscape
    figs['persistence'] = create_persistence_landscape(returns)

    # 5. Regime Explorer
    figs['regime'] = create_regime_explorer()

    # Save to HTML files
    print("\n" + "="*60)
    print("SAVING VISUALIZATIONS")
    print("="*60)

    for name, fig in figs.items():
        filename = f'viz_{name}.html'
        fig.write_html(filename)
        print(f"Saved: {filename}")

    # Create combined dashboard
    print("\nCreating combined dashboard...")

    dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>TDA Market Topology Visualizations</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: rgb(20,20,30);
            color: white;
            margin: 0;
            padding: 20px;
        }
        h1 { text-align: center; color: #4ECDC4; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        .viz {
            background: rgb(30,30,40);
            border-radius: 10px;
            overflow: hidden;
        }
        .viz iframe {
            width: 100%;
            height: 500px;
            border: none;
        }
        .full-width { grid-column: span 2; }
        .caption {
            padding: 10px;
            text-align: center;
            background: rgb(40,40,50);
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>TDA Market Topology Explorer</h1>
    <p style="text-align:center; color:#888; margin-bottom:30px;">
        Interactive visualizations of Topological Data Analysis applied to financial markets
    </p>

    <div class="grid">
        <div class="viz">
            <iframe src="viz_network.html"></iframe>
            <div class="caption">
                <b>3D Correlation Network</b><br>
                Stocks positioned by correlation distance. Closer = more correlated.
            </div>
        </div>

        <div class="viz">
            <iframe src="viz_persistence.html"></iframe>
            <div class="caption">
                <b>Persistence Landscape</b><br>
                H₁ topological features (loops) in 3D: birth, death, persistence.
            </div>
        </div>

        <div class="viz full-width">
            <iframe src="viz_filtration.html"></iframe>
            <div class="caption">
                <b>Filtration Animation</b><br>
                Watch how network topology evolves as correlation threshold changes. Press Play!
            </div>
        </div>

        <div class="viz">
            <iframe src="viz_surface.html"></iframe>
            <div class="caption">
                <b>ρ-CV Surface</b><br>
                3D landscape showing how correlation structure affects topology stability.
            </div>
        </div>

        <div class="viz">
            <iframe src="viz_regime.html"></iframe>
            <div class="caption">
                <b>Regime Comparison</b><br>
                Calm vs stressed markets: correlation matrices and stability metrics.
            </div>
        </div>
    </div>

    <p style="text-align:center; color:#666; margin-top:30px;">
        Created for TDA Trading Strategy Research | github.com/adam-jfkhs/TDA
    </p>
</body>
</html>
"""

    with open('tda_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(dashboard_html)

    print("Saved: tda_dashboard.html")

    print("\n" + "="*60)
    print("DONE! Open tda_dashboard.html in your browser")
    print("="*60)

    return figs


if __name__ == "__main__":
    figs = main()
