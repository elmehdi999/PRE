import numpy as np
import os
import sys
import warnings
from sklearn.linear_model import OrthogonalMatchingPursuitCV, OrthogonalMatchingPursuit

from nisp_rff import NISP_RFF

warnings.filterwarnings("ignore")

def diagnostic_claude():
    fichier_cache = "Y_eval_full_D50_N2048.npy"
    if not os.path.exists(fichier_cache):
        print(f"Fichier {fichier_cache} introuvable.")
        sys.exit(1)
        
    data = np.load(fichier_cache, allow_pickle=True).item()
    Y_eval_full = data['Y_eval_full']
    Xi = data['Xi']
    N_evals = Y_eval_full.shape[0]
    
    ref_mean = np.load("mc_mean_100k.npy")
    ref_var = np.load("mc_var_100k.npy")
    
    nisp = NISP_RFF(50, 3, N_evals)
    nisp.Xi = Xi
    nisp.generer_multi_indices(q=0.75)
    nisp.evaluer_polynomes_hermite()
    Psi = nisp.Psi
    
    # 1. Comparaison locale au noeud chaud
    hot_idx = np.argmax(ref_mean)
    var_mc_chaud = ref_var[hot_idx]
    
    Y_chaud = Y_eval_full[:, hot_idx]
    omp_cv = OrthogonalMatchingPursuitCV(fit_intercept=False, cv=5)
    omp_cv.fit(Psi, Y_chaud)
    
    var_omp_chaud = np.sum(omp_cv.coef_[1:]**2)
    erreur_rel_chaud = abs(var_omp_chaud - var_mc_chaud) / var_mc_chaud
    
    print("\n" + "-"*50)
    print(" 1. DIAGNOSTIC LOCAL (NOEUD CHAUD)")
    print("-"*50)
    print(f"Variance MC 100k : {var_mc_chaud:.4f}")
    print(f"Variance OMP     : {var_omp_chaud:.4f}")
    print(f"Ecart relatif    : {erreur_rel_chaud*100:.2f} %")
    
    # 2. OMP Multi-sorties global
    n_coefs = 60
    print("\n" + "-"*50)
    print(f" 2. DIAGNOSTIC GLOBAL (OMP MULTI-SORTIES, {n_coefs} coefs)")
    print("-"*50)
    
    omp_multi = OrthogonalMatchingPursuit(n_nonzero_coefs=n_coefs, fit_intercept=False)
    omp_multi.fit(Psi, Y_eval_full)
    
    C_full = omp_multi.coef_
    
    PCE_mean_field = C_full[:, 0]
    PCE_var_field = np.sum(C_full[:, 1:]**2, axis=1)
    
    err_m = np.linalg.norm(PCE_mean_field - ref_mean) / np.linalg.norm(ref_mean)
    err_v = np.linalg.norm(PCE_var_field - ref_var) / np.linalg.norm(ref_var)
    
    print(f"Erreur L2 Moyenne analytique  : {err_m*100:.2f} %")
    print(f"Erreur L2 Variance analytique : {err_v*100:.2f} %")
    
    nisp.finalise()

if __name__ == "__main__":
    diagnostic_claude()