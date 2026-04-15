# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    plt.rcParams['text.usetex'] = False
    data_dir = 'data/'
    filepath = os.path.join(data_dir, 'cleaned_perovskite_data.csv')
    df = pd.read_csv(filepath)
    df_uncharacterized = df[df['K_VRH'].isna()].copy()
    df_uncharacterized = df_uncharacterized.dropna(subset=['predicted_is_stable_prob', 'predicted_pugh_ratio_std'])
    df_uncharacterized = df_uncharacterized.reset_index(drop=True)
    obj1 = df_uncharacterized['predicted_is_stable_prob'].values
    obj2 = df_uncharacterized['predicted_pugh_ratio_std'].values
    band_gap = df_uncharacterized['predicted_band_gap'].values
    formulas = df_uncharacterized['formula'].values
    n = len(obj1)
    pareto_front = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if (obj1[j] >= obj1[i] and obj2[j] <= obj2[i]) and (obj1[j] > obj1[i] or obj2[j] < obj2[i]):
                dominated = True
                break
        if not dominated:
            pareto_front.append(i)
    pareto_front = np.array(pareto_front)
    pareto_sorted_indices = pareto_front[np.argsort(obj1[pareto_front])[::-1]]
    top_10_indices = pareto_sorted_indices[:10]
    print('Total uncharacterized materials analyzed: ' + str(n))
    print('Number of Pareto-optimal candidates found: ' + str(len(pareto_front)))
    print('\nTop 10 Pareto-optimal candidates (sorted by predicted stability probability):')
    print('-' * 80)
    print('Formula         | Stability Prob  | Mech Uncertainty   | Band Gap (eV)  ')
    print('-' * 80)
    for idx in top_10_indices:
        form = str(formulas[idx])
        stab = str(round(obj1[idx], 4))
        unc = str(round(obj2[idx], 4))
        bg = str(round(band_gap[idx], 4))
        print(form.ljust(15) + ' | ' + stab.ljust(15) + ' | ' + unc.ljust(18) + ' | ' + bg.ljust(15))
    print('-' * 80 + '\n')
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(obj2, obj1, c=band_gap, cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.5, s=40)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Predicted Band Gap (eV)')
    pareto_plot_indices = pareto_front[np.argsort(obj2[pareto_front])]
    ax.plot(obj2[pareto_plot_indices], obj1[pareto_plot_indices], color='red', linestyle='--', linewidth=2, label='Pareto Front')
    for i, idx in enumerate(top_10_indices):
        y_offset = 15 if i % 2 == 0 else -20
        x_offset = 15 if (i // 2) % 2 == 0 else -20
        ax.annotate(formulas[idx], (obj2[idx], obj1[idx]), xytext=(x_offset, y_offset), textcoords='offset points', fontsize=9, color='black', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8), arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    ax.set_xlabel('Mechanical Uncertainty (Predicted Pugh Ratio Std)')
    ax.set_ylabel('Predicted Stability Probability')
    ax.set_title('Pareto Optimization: Stability vs. Mechanical Uncertainty')
    if obj2.max() > 100 * max(obj2.min(), 1e-05):
        ax.set_xscale('log')
    ax.legend()
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = 'pareto_front_1_' + timestamp + '.png'
    plot_filepath = os.path.join(data_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print('Pareto front plot saved to ' + plot_filepath)

if __name__ == '__main__':
    main()