import numpy as np
import sys
import subprocess
import re

def tester_un_K(k_val):
    script = f"""
import sys, numpy as np, petsc4py
from mefpp4py import mefpp
petsc4py.init(sys.argv)
mefpp.initialise("carre")
mefpp.litEtExecuteActionsDansCollection()
gfc = mefpp.reqCollectionDeCorps().reqCorps("carre").reqGFC()

vec_T = gfc.reqVecteurPETSc("T_imp").reqVec()
vec_Texacte = gfc.reqVecteurPETSc("T_exacte_vec").reqVec()

indices_T = np.arange(vec_T.getSize(), dtype=np.int32)

# 1. Solution mathématique de référence
gfc.reqPP("pp_interpoleTexacte").execute()
gfc.reqPP("pp_copie_Texacte_vec").execute()
T_reference = vec_Texacte.getValues(indices_T).copy()

# 2. Injection stricte de K via l'interpréteur MEF++ 
# (Ceci bypasse totalement le bug du vecteur K_imp nodal vs K élémentaire)
gfc.lireLigne(f'scalaire K_valeur_test f(x,y,z)={k_val}')
gfc.lireLigne('pp_reinterpole pp_force_K [K, K_valeur_test]')
gfc.reqPP("pp_force_K").execute()

# 3. Résolution physique avec le solveur exact (LU)
gfc.reqPP("ppAssMatEtRes").execute()
gfc.reqPP("resolution").execute()
gfc.reqPP("pp_copie_T_imp").execute()

# 4. Extraction et Comparaison
T_calcule = vec_T.getValues(indices_T).copy()
T_theorique = T_reference / {k_val}

T_calc_max = np.max(T_calcule)
T_theo_max = np.max(T_theorique)
erreur_max = np.max(np.abs(T_calcule - T_theorique))

print(f"RESULTAT K={k_val} Tcalc={{T_calc_max:.6f}} Ttheo={{T_theo_max:.6f}} erreur_max={{erreur_max:.6e}}")
mefpp.finalise()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    return result.stdout, result.stderr

if __name__ == "__main__":
    valeurs_K = [1.0, 2.0, 0.5, 10.0]
    print("Validation numérique vs analytique (process isolés) :")
    tous_ok = True
    
    for k_val in valeurs_K:
        stdout, stderr = tester_un_K(k_val)
        
        # Extraction sécurisée des valeurs brutes via Regex
        match = re.search(r"RESULTAT K=[\d\.]+ Tcalc=([\d\.\-]+) Ttheo=([\d\.\-]+) erreur_max=([0-9\.\-eE\+]+)", stdout)
        
        if match:
            t_calc = float(match.group(1))
            t_theo = float(match.group(2))
            err = float(match.group(3))
            
            print(f"K={k_val:<4} : T_calc = {t_calc:.4f} | T_theo = {t_theo:.4f} | Erreur = {err:.4e}")
            
            if err > 1e-2:
                print(f"  [!] ÉCART DÉTECTÉ (Tolérance 1e-2 dépassée)")
                tous_ok = False
        else:
            print(f"K={k_val:<4} : ÉCHEC DE LECTURE — {stderr[-300:]}")
            tous_ok = False
            
    print("\n==================================================")
    print("=> TEST GLOBAL:", "PASSÉ" if tous_ok else "ÉCHOUÉ (Vérifier si dû à la discrétisation du maillage)")
    print("==================================================")