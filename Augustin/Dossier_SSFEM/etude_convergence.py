import subprocess
import matplotlib.pyplot as plt
import re
import sys

def run_convergence():
    D = 50
    P = 3
    liste_N = [2**n for n in range(3,12)]
    
    erreurs_moyenne = []
    erreurs_variance = []
    
    print("="*60)
    print(f" DÉMARRAGE ÉTUDE DE CONVERGENCE (D={D}, P={P})")
    print("="*60)
    
    for N in liste_N:
        print(f"\n---> Lancement d'un processus isolé pour N = {N} <---")
        
        # Appel du script en tant que sous-processus indépendant pour purger la mémoire C++
        # équivalent à taper "python nisp_rff.py 50 3 N" dans le terminal
        result = subprocess.run(
            [sys.executable, "nisp_rff.py", str(D), str(P), str(N)], 
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print(f"ERREUR lors de l'exécution pour N={N} :")
            print(result.stderr)
            sys.exit(1)
            
        # Extraction des erreurs L2 via expressions régulières (Regex)
        match_moy = re.search(r"Erreur L2 sur la Moyenne\s*:\s*([0-9\.eE+-]+)", result.stdout)
        match_var = re.search(r"Erreur L2 sur la Variance\s*:\s*([0-9\.eE+-]+)", result.stdout)
        
        if match_moy and match_var:
            err_m = float(match_moy.group(1))
            err_v = float(match_var.group(1))
            erreurs_moyenne.append(err_m)
            erreurs_variance.append(err_v)
            print(f" Succès : Erreur Moyenne = {err_m*100:.3f} % | Erreur Variance = {err_v*100:.2f} %")
        else:
            print(f"ERREUR : Impossible de lire les erreurs L2 dans la sortie pour N={N}.")
            print("Dernières lignes du terminal caché :")
            print("\n".join(result.stdout.split("\n")[-20:]))
            sys.exit(1)

    # --- TRACÉ DE LA COURBE DE CONVERGENCE ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(liste_N, erreurs_moyenne, 'o-', color='blue', linewidth=2, label='Erreur L2 relative (Moyenne)')
    plt.plot(liste_N, erreurs_variance, 's-', color='red', linewidth=2, label='Erreur L2 relative (Variance)')
    
    # Échelle logarithmique indispensable pour voir la convergence
    plt.yscale('log')
    plt.xscale('log', base=2)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel("Nombre d'évaluations MEF++ ($N$)", fontsize=12)
    plt.ylabel("Erreur Spatiale Globale $L_2$", fontsize=12)
    plt.title(f"Convergence Spatiale NISP-RFF (D={D}, P={P}, q=0.75)", fontsize=14)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("convergence_N.png", dpi=300)
    print("\n[SUCCÈS] Courbe générée avec succès : convergence_N.png")

if __name__ == "__main__":
    run_convergence()