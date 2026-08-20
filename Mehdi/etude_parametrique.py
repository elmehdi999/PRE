import subprocess
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import re

def lancer_balayage():
    print("="*60)
    print(" BALAYAGE PARAMÉTRIQUE : RECHERCHE DE L'OPTIMUM L_CORR")
    print("="*60)

    # Paramètres de l'étude (N=1024)
    D = 30
    P = 3
    N = 1024
    
    valeurs_lcorr = [0.06, 0.08, 0.10, 0.12, 0.14, 0.17, 0.20, 0.25]
    erreurs_variance = []
    
    # Détection exhaustive du fichier de référence pour vérification initiale
    ref_found = False
    for ref_file in ["mc_var_100k.npy", "mc_var_10k.npy", "mc_var_1000.npy", "variance_mc_spatial.npy"]:
        if os.path.exists(ref_file):
            print(f"-> Référence trouvée pour la validation L2 : {ref_file}")
            ref_found = True
            break
            
    if not ref_found:
        print("Erreur : Aucun fichier de référence variance introuvable dans le dossier.")
        sys.exit(1)

    for l_corr in valeurs_lcorr:
        print(f"\n---> Lancement du sous-processus pour l_corr = {l_corr:.4f} <---")
        
        # Appel propre via subprocess pour isoler MEF++ dans un processus jetable
        result = subprocess.run(
            [sys.executable, "nisp_rff.py", str(D), str(P), str(N), str(l_corr)],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print(f" [ERREUR] Le test pour l_corr={l_corr} a planté.")
            print(result.stderr)
            sys.exit(1)
            
        # Extraction du résultat via une expression régulière
        match = re.search(r"Erreur L2 sur la Variance\s*:\s*[0-9\.eE+-]+\s*\(\s*([0-9\.]+)\s*%\)", result.stdout)
        
        if match:
            err_var_pct = float(match.group(1))
            erreurs_variance.append(err_var_pct)
            print(f" [SUCCÈS] Erreur Spatiale Variance = {err_var_pct:.2f}%")
        else:
            print(f" [ERREUR] Impossible de lire l'erreur dans le terminal pour l_corr={l_corr}.")
            print(" Dernière lignes du terminal :")
            print("\n".join(result.stdout.split("\n")[-10:]))
            sys.exit(1)

    # --- TRACÉ DE LA COURBE DE LA VALLÉE ---
    os.makedirs("resultats", exist_ok=True)
    plt.figure(figsize=(9, 6))
    
    plt.plot(valeurs_lcorr, erreurs_variance, 's-', color='crimson', linewidth=2.5, markersize=8)
    
    # On identifie le point le plus bas
    idx_min = np.argmin(erreurs_variance)
    l_corr_opt = valeurs_lcorr[idx_min]
    err_min = erreurs_variance[idx_min]
    
    plt.plot(l_corr_opt, err_min, 'k*', markersize=12, label=f'Minimum empirique ({l_corr_opt:.2f}, {err_min:.1f}%)')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("Longueur de corrélation spatiale RFF ($l_{corr}$)", fontsize=12)
    plt.ylabel("Erreur Spatiale Globale $L_2$ sur la Variance (%)", fontsize=12)
    plt.title(f"Recherche de l'optimum géométrique (D={D}, P={P}, N={N})", fontsize=14)
    plt.legend(fontsize=12)
    
    nom_fig = "resultats/Sweep_Lcorr_Variance.png"
    plt.tight_layout()
    plt.savefig(nom_fig, dpi=300)
    print("\n" + "="*60)
    print(f" BALAYAGE TERMINÉ. L'optimum mathématique observé est l_corr = {l_corr_opt:.4f}")
    print(f" Graphique sauvegardé : {nom_fig}")
    print("="*60)

if __name__ == "__main__":
    lancer_balayage()
