# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import datetime

def main():
    data_dir = "data/"
    input_file = os.path.join(data_dir, "classification_results.csv")
    df = pd.read_csv(input_file)
    df['is_stable'] = df['is_stable'].astype(bool)
    df['is_silicate'] = (df['B_site'] == 'Si') | (df['A_site'] == 'Si')
    silicates = df[df['is_silicate']]
    non_silicates = df[~df['is_silicate']]
    print("Silicate Paradox and Pressure-Volume Analysis\n" + "="*45)
    print("Total silicates: " + str(len(silicates)))
    print("Total non-silicates: " + str(len(non_silicates)))
    if len(silicates) > 1:
        r_sil, p_sil = pearsonr(silicates['volume_residual'], silicates['energy_above_hull'])
    else:
        r_sil, p_sil = np.nan, np.nan
    if len(non_silicates) > 1:
        r_nsil, p_nsil = pearsonr(non_silicates['volume_residual'], non_silicates['energy_above_hull'])
    else:
        r_nsil, p_nsil = np.nan, np.nan
    print("\nPearson correlation between volume_residual and energy_above_hull:")
    print("Silicates: r = " + str(round(r_sil, 4)) + ", p-value = " + str(p_sil))
    print("Non-silicates: r = " + str(round(r_nsil, 4)) + ", p-value = " + str(p_nsil))
    if len(silicates) > 1:
        r_sil_prob, p_sil_prob = pearsonr(silicates['volume_residual'], silicates['stability_probability'])
    else:
        r_sil_prob, p_sil_prob = np.nan, np.nan
    if len(non_silicates) > 1:
        r_nsil_prob, p_nsil_prob = pearsonr(non_silicates['volume_residual'], non_silicates['stability_probability'])
    else:
        r_nsil_prob, p_nsil_prob = np.nan, np.nan
    print("\nPearson correlation between volume_residual and stability_probability:")
    print("Silicates: r = " + str(round(r_sil_prob, 4)) + ", p-value = " + str(p_sil_prob))
    print("Non-silicates: r = " + str(round(r_nsil_prob, 4)) + ", p-value = " + str(p_nsil_prob))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax1 = axes[0]
    stable_sil = silicates[silicates['is_stable'] == True]
    unstable_sil = silicates[silicates['is_stable'] == False]
    if len(unstable_sil) > 0:
        ax1.scatter(unstable_sil['volume_residual'], unstable_sil['energy_above_hull'], color='red', alpha=0.6, label='Metastable/Unstable', edgecolors='k')
    if len(stable_sil) > 0:
        ax1.scatter(stable_sil['volume_residual'], stable_sil['energy_above_hull'], color='blue', alpha=0.8, label='Stable', edgecolors='k')
    ax1.set_title('Silicate Perovskites')
    ax1.set_xlabel('Volume Residual (Å³)')
    ax1.set_ylabel('Energy Above Hull (eV/atom)')
    ax1.set_xscale('symlog', linthresh=150)
    text_sil = "r = " + str(round(r_sil, 2)) + "\np = " + str(p_sil)
    ax1.text(0.05, 0.95, text_sil, transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    if len(silicates) > 0:
        ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2 = axes[1]
    stable_nsil = non_silicates[non_silicates['is_stable'] == True]
    unstable_nsil = non_silicates[non_silicates['is_stable'] == False]
    if len(unstable_nsil) > 0:
        ax2.scatter(unstable_nsil['volume_residual'], unstable_nsil['energy_above_hull'], color='red', alpha=0.6, label='Metastable/Unstable', edgecolors='k')
    if len(stable_nsil) > 0:
        ax2.scatter(stable_nsil['volume_residual'], stable_nsil['energy_above_hull'], color='blue', alpha=0.8, label='Stable', edgecolors='k')
    ax2.set_title('Non-Silicate Perovskites')
    ax2.set_xlabel('Volume Residual (Å³)')
    ax2.set_xscale('symlog', linthresh=150)
    text_nsil = "r = " + str(round(r_nsil, 2)) + "\np = " + str(p_nsil)
    ax2.text(0.05, 0.95, text_nsil, transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    if len(non_silicates) > 0:
        ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join(data_dir, 'silicate_paradox_4_' + timestamp + '.png')
    plt.savefig(plot_filename, dpi=300)
    print("\nScatter plot saved to " + plot_filename)

if __name__ == '__main__':
    main()