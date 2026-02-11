import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family':       ['Noto Sans CJK JP', 'DejaVu Sans'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.18,
    'grid.linestyle':    ':',
    'grid.linewidth':    0.5,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
})

C_BASE = '#4472A8'
C_DCOT = '#C0503A'
SZ = 95

def mk(temp, prompt):
    if prompt == 'NP': return 's'
    return 'D' if temp == 'Dynamic' else 'o'

def col(model):
    return C_DCOT if model == 'LoRA' else C_BASE

MMLU = [
    ('BLN',  'Base', 'Lock',    'NP', 55.66, 1741.6),
    ('BLC',  'Base', 'Lock',    'UP', 58.24, 1595.0),
    ('BDC',  'Base', 'Dynamic', 'UP', 58.24, 1618.0),
    ('DLN',  'LoRA', 'Lock',    'NP', 64.73, 1495.5),
    ('DLC',  'LoRA', 'Lock',    'UP', 62.92, 1198.9),
    ('DDC',  'LoRA', 'Dynamic', 'UP', 63.43, 1201.9),
]
GPQA = [
    ('BLN',  'Base', 'Lock',    'NP', 43.03, 5875.1),
    ('BLC',  'Base', 'Lock',    'UP', 45.05, 5538.9),
    ('BDC',  'Base', 'Dynamic', 'UP', 45.05, 5630.0),
    ('DLN',  'LoRA', 'Lock',    'NP', 52.83, 2772.3),
    ('DLC',  'LoRA', 'Lock',    'UP', 51.41, 2073.2),
    ('DDC',  'LoRA', 'Dynamic', 'UP', 52.93, 2153.4),
]

SHORT = {
    'BLN': 'Base\nT.Locked/Base',
    'BLC': 'Base\nT.Locked/Custom',
    'BDC': 'Base\nT.Dynamic/Custom',
    'DLN': 'D-CoT\nT.Locked/Base',
    'DLC': 'D-CoT\nT.Locked/Custom',
    'DDC': 'D-CoT\nT.Dynamic/Custom',
}

# (dx, dy, ha)
ANN_M = {
    'BLN': (  28,  0.10, 'left'),
    'BLC': (  18,  0.42, 'left'),
    'BDC': (  18, -1.15, 'left'),
    'DLN': (  18,  0.20, 'left'),
    'DLC': (  18, -1.10, 'left'),
    'DDC': (  18,  0.55, 'left'),
}
ANN_G = {
    'BLN': ( 150, -1.42, 'left'),
    'BLC': ( 150,  0.42, 'left'),
    'BDC': ( 150, -1.42, 'left'),
    'DLN': ( 180, -0.20, 'left'),
    'DLC': ( 180, -1.15, 'left'),
    'DDC': ( 180,  1.50, 'left'),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
fig.patch.set_facecolor('#FAFAFA')
fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.22, wspace=0.38)

def plot_panel(ax, data, ann, x_div, xlabel, title, xlim, ylim):
    for k, model, temp, prompt, acc, tok in data:
        ax.scatter(tok/x_div, acc,
                   c=col(model), marker=mk(temp, prompt), s=SZ,
                   edgecolors='white', linewidths=0.9, zorder=4)
    for k, model, temp, prompt, acc, tok in data:
        dx, dy, ha = ann[k]
        ax.annotate(SHORT[k], (tok/x_div, acc),
                    xytext=(tok/x_div + dx, acc + dy),
                    fontsize=7.8, color='#222', ha=ha,
                    arrowprops=dict(arrowstyle='-', color='#ccc', lw=0.6))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left', pad=8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_facecolor('#FAFAFA')

plot_panel(ax1, MMLU, ANN_M, 1,
           'Average Output Tokens',
           '(A)  MMLU-Pro  (0-shot, 12k)',
           (1050, 1950), (54.0, 67.0))

plot_panel(ax2, GPQA, ANN_G, 1,
           'Average Output Tokens',
           '(B)  GPQA-Diamond  (0-shot, 5-seed avg)',
           (1400, 7300), (41.0, 56.0))

handles = [
    Line2D([0],[0], marker='s', color=C_BASE, linestyle='None', markersize=8,
           markeredgecolor='white', markeredgewidth=0.8,
           label='Baseline / Temp. Locked / Base Prompt'),
    Line2D([0],[0], marker='o', color=C_BASE, linestyle='None', markersize=8,
           markeredgecolor='white', markeredgewidth=0.8,
           label='Baseline / Temp. Locked / Custom Prompt'),
    Line2D([0],[0], marker='D', color=C_BASE, linestyle='None', markersize=8,
           markeredgecolor='white', markeredgewidth=0.8,
           label='Baseline / Temp. Dynamic / Custom Prompt'),
    Line2D([0],[0], marker='s', color=C_DCOT, linestyle='None', markersize=8,
           markeredgecolor='white', markeredgewidth=0.8,
           label='D-CoT / Temp. Locked / Base Prompt'),
    Line2D([0],[0], marker='o', color=C_DCOT, linestyle='None', markersize=8,
           markeredgecolor='white', markeredgewidth=0.8,
           label='D-CoT / Temp. Locked / Custom Prompt'),
    Line2D([0],[0], marker='D', color=C_DCOT, linestyle='None', markersize=8,
           markeredgecolor='white', markeredgewidth=0.8,
           label='D-CoT / Temp. Dynamic / Custom Prompt'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8.5,
           framealpha=0.95, columnspacing=1.4, handlelength=1.4,
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle('D-CoT: 正答率と出力トークン数の比較  (Qwen3-8B, 0-shot)',
             fontsize=12, fontweight='bold', y=0.97, color='#111')

out = '/mnt/user-data/outputs/dcot_scatter_2panel.png'
plt.savefig(out, dpi=220, bbox_inches='tight', facecolor=fig.get_facecolor())
print('Saved ->', out)
plt.close()
