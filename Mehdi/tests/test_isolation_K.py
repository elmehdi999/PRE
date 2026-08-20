import subprocess
import sys
import re

def run_isolation_test():
    print("="*60)
    print(" TEST D'ISOLATION STRICTE DE L'ÉTAT DU SOLVEUR (K * T)")
    print("="*60)
    
    k_tests = [0.10, 0.19, 0.50, 1.00, 10.00]
    
    for k in k_tests:
        # Lancement dans un processus totalement vierge
        result = subprocess.run(
            [sys.executable, "test_single_K.py", str(k)],
            capture_output=True, text=True
        )
        
        match = re.search(r"RESULTAT K=[\d\.]+ T_noeud97=([\d\.]+)", result.stdout)
        if match:
            t = float(match.group(1))
            print(f" K={k:<5.2f} | T[97] (process isolé) = {t:<9.6f} | K * T[97] = {k*t:.6f}")
        else:
            print(f" K={k:<5.2f} : Échec de l'extraction.")
            print("Détail de l'erreur stderr :")
            print(result.stderr[-500:])
            
    print("\n-> Si K * T[97] est strictement constant (ex: 8.46), la non-linéarité précédente")
    print("   était une pure pollution d'état (Hypothèse A confirmée).")
    print("-> Sinon, c'est une physique interne au .dat (Hors périmètre perturbatif NISP).")

if __name__ == "__main__":
    run_isolation_test()