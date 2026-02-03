import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Plotting Style configuration
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'lines.linewidth': 2.2,
    'axes.linewidth': 1,
    'grid.linewidth': 0.7,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8F8F8',
    'grid.alpha': 0.4,
})

MODELS = ["small.en", "medium.en", "distil-large-v3", "large-v3-turbo"]
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
MODEL_COLORS = dict(zip(MODELS, COLORS))

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["environment"] = (
                        df["cpu_model"] + " (" + 
                        df["cpu_cores_logical"].astype(str) + "T/" + 
                        df["cpu_cores_physical"].astype(str) + "C) " +
                        df["cpu_arch"]
                     )
    return df.sort_values(["environment", "model", "threads"])

def plot_by_environment(df, y_col, y_label, title, output_path, add_rtf_line=False, add_linear_scaling=False):
    envs = df["environment"].unique()
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 5), sharey=True)
    axes = [axes] if len(envs) == 1 else axes
    
    for ax, env in zip(axes, envs):
        sub = df[df["environment"] == env]
        
        for model in MODELS:
            m = sub[sub["model"] == model]
            if not m.empty:
                ax.plot(m["threads"], m[y_col], marker='o', markersize=6,
                       label=model, color=MODEL_COLORS[model])
        
        # Reference lines
        physical_cores = int(sub["cpu_cores_physical"].iloc[0])
        ax.axvline(physical_cores, color='#555', linestyle='--', linewidth=1.5,
                  label=f'Physical cores ({physical_cores})', alpha=0.6)
        
        if add_rtf_line:
            ax.axhline(1.0, color='#777', linestyle=':', linewidth=1.5,
                      label='Real-time', alpha=0.6)

        if add_linear_scaling:
            max_threads = sub["threads"].max()
            ax.plot([1, max_threads], [1, max_threads], 
                   color='#999', linestyle='--', linewidth=1.5,
                   label='Linear scaling', alpha=0.5, zorder=1)
        
        ax.set_title(env, fontweight='500', pad=12)
        ax.set_xlabel('Threads', fontweight='500')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_axisbelow(True)
    
    axes[0].set_ylabel(y_label, fontweight='500')
    for ax in axes:
        ax.legend(framealpha=0.95, edgecolor='#CCC')
    
    fig.suptitle(title, fontweight='600')
    fig.tight_layout()
    fig.savefig(output_path, facecolor='white')
    plt.close()
    print(f"Saved {output_path}")

def plot_rtf_vs_threads(csv_path, output_path):
    df = load_data(csv_path)
    plot_by_environment(df, "rtf", "Real-Time Factor (lower is better)",
                       "RTF vs Threads (CPU-only)", output_path, add_rtf_line=True)

def plot_best_rtf(csv_path, output_path):
    df = load_data(csv_path)
    summary = df.groupby(["environment", "model"]).apply(
        lambda g: g.loc[g["rtf"].idxmin()]
    ).reset_index(drop=True)
    
    envs = summary["environment"].unique()
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 5), sharey=True)
    axes = [axes] if len(envs) == 1 else axes
    
    for ax, env in zip(axes, envs):
        sub = summary[summary["environment"] == env].sort_values('rtf')
        
        bars = ax.barh(sub["model"], sub["rtf"], height=0.6,
                      color=[MODEL_COLORS[m] for m in sub["model"]], alpha=0.85)
        
        for idx, row in sub.iterrows():
            ax.text(row["rtf"] * 1.02, row["model"],
                   f' {row["rtf"]:.2f} ({int(row["threads"])}t)',
                   va='center', fontsize=10)
        
        ax.axvline(1.0, color='#777', linestyle=':', linewidth=1.5, alpha=0.6)
        ax.set_title(env, fontweight='500', pad=12)
        ax.set_xlabel('Best RTF (lower is better)', fontweight='500')
        ax.grid(True, axis='x', linestyle=':', alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(0, sub['rtf'].max() * 1.25)
    
    fig.suptitle('Best RTF per Model (CPU-only)', fontweight='600')
    fig.tight_layout()
    fig.savefig(output_path, facecolor='white')
    plt.close()
    print(f"Saved {output_path}")

def plot_rtf_efficiency(csv_path, output_path):
    df = load_data(csv_path)
    df["rtf_1t"] = df.groupby(["environment", "model"])["rtf"].transform("first")
    df["efficiency"] = df["rtf_1t"] / df["rtf"]
    
    plot_by_environment(df, "efficiency", "Speedup vs 1 Thread",
                       "Multi-Threading Efficiency", output_path, add_linear_scaling=True)

if __name__ == "__main__":
    DATA_DIR = Path("data")
    CSV_PATH = DATA_DIR / "faster_whisper_cpu_benchmarks.csv"
    
    plot_rtf_vs_threads(CSV_PATH, DATA_DIR / "rtf_vs_threads.png")
    plot_best_rtf(CSV_PATH, DATA_DIR / "best_rtf.png")
    plot_rtf_efficiency(CSV_PATH, DATA_DIR / "rtf_efficiency.png")