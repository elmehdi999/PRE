import subprocess
import re
import sys
import matplotlib.pyplot as plt

def lancer_etude():
    D = 50
    P = 3
    # On teste l'évolution de la précision en fonction de l'apprentissage
    liste_N = [2**n for n in range(3,14)]
    
    erreurs_moyenne = []
    erreurs_variance = []
    
    print("="*60)
    print(f" DÉMARRAGE DE L'ÉTUDE DE CONVERGENCE NISP-RFF (D={D}, P={P})")
    print("="*60)
    
    for N in liste_N:
        print(f"\n---> Lancement du test pour N = {N} échantillons <---")
        
        # On lance nisp_rff.py comme un processus indépendant
        result = subprocess.run(
            [sys.executable, "nisp_rff.py", str(D), str(P), str(N)], 
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print(f"ERREUR lors de l'exécution pour N={N}.")
            print("Détail de l'erreur :")
            print(result.stderr)
            sys.exit(1)
            
        # Extraction des pourcentages d'erreur via Regex
        match_moy = re.search(r"Erreur L2 sur la Moyenne\s*:\s*[0-9\.eE+-]+\s*\(\s*([0-9\.]+)\s*%\)", result.stdout)
        match_var = re.search(r"Erreur L2 sur la Variance\s*:\s*[0-9\.eE+-]+\s*\(\s*([0-9\.]+)\s*%\)", result.stdout)
        
        if match_moy and match_var:
            err_m = float(match_moy.group(1))
            err_v = float(match_var.group(1))
            erreurs_moyenne.append(err_m)
            erreurs_variance.append(err_v)
            print(f" [SUCCÈS N={N}] Erreur Moyenne: {err_m:.2f}% | Erreur Variance: {err_v:.2f}%")
        else:
            print(f"Erreur de lecture du terminal pour N={N}.")
            print(result.stdout[-500:])
            sys.exit(1)

    # --- TRACÉ DE LA COURBE DE CONVERGENCE ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(liste_N, erreurs_moyenne, 'o-', color='blue', linewidth=2, label='Erreur L2 (Moyenne)')
    plt.plot(liste_N, erreurs_variance, 's-', color='red', linewidth=2, label='Erreur L2 (Variance)')
    
    plt.yscale('log') # Échelle log indispensable pour voir la chute de l'erreur
    plt.xscale('log', base=2)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel("Nombre d'évaluations MEF++ ($N$)", fontsize=12)
    plt.ylabel("Erreur Spatiale Globale $L_2$ (%)", fontsize=12)
    plt.title(f"Convergence Spatiale NISP-RFF (D={D}, P={P})", fontsize=14)
    plt.legend(fontsize=12)
    
    nom_image = "convergence_N.png"
    plt.tight_layout()
    plt.savefig(nom_image, dpi=300)
    print("\n" + "="*60)
    print(f" FIN DE L'ÉTUDE. Courbe générée avec succès : {nom_image}")
    print("="*60)

if __name__ == "__main__":
    lancer_etude()