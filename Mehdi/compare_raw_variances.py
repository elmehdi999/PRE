import numpy as np
import os

def comparer_variances():
    print("="*50)
    print(" EXTRACTION DES VARIANCES BRUTES (NOEUD CHAUD)")
    print("="*50)

    # 1. Chargement de la référence MC
    if not os.path.exists("mc_mean_10k.npy") or not os.path.exists("mc_var_10k.npy"):
        print("Erreur : mc_var_10k.npy introuvable.")
        return
    
    mc_mean = np.load("mc_mean_10k.npy")
    mc_var = np.load("mc_var_10k.npy")
    noeud_chaud = int(np.argmax(mc_mean))

    # 2. Chargement des données NISP
    data_nisp = np.load("Y_eval_full_D30_N3000.npy", allow_pickle=True).item()
    Y_full = data_nisp['Y_eval_full']
    var_nisp = np.var(Y_full, axis=0, ddof=1)

    print(f"Noeud Chaud identifié : {noeud_chaud}")
    print(f"-> Variance brute Monte Carlo : {mc_var[noeud_chaud]:.6f}")
    print(f"-> Variance brute NISP-RFF    : {var_nisp[noeud_chaud]:.6f}")
    
    # Rappel de la variance SSFEM obtenue dans ton terminal précédent
    print(f"-> Variance brute SSFEM       : 0.155000 (Issue du run SSFEM_SciPy D=30, P=5)")
    
    diff_nisp = var_nisp[noeud_chaud] - mc_var[noeud_chaud]
    diff_ssfem = 0.1550 - mc_var[noeud_chaud]
    
    print("\nAnalyse de direction :")
    print(f" NISP  s'écarte de {diff_nisp:+.6f}")
    print(f" SSFEM s'écarte de {diff_ssfem:+.6f}")

if __name__ == "__main__":
    comparer_variances()