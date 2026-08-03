import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
import os
import math
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc

def E_He3_normalized(a, b, c):
    if (a + b + c) % 2 != 0: return 0.0
    s = (a + b + c) // 2
    if s < a or s < b or s < c: return 0.0
    num = math.sqrt(math.factorial(a) * math.factorial(b) * math.factorial(c))
    den = math.factorial(s - a) * math.factorial(s - b) * math.factorial(s - c)
    return num / den

def calcul_cijk_tuple(alpha_i, alpha_j, alpha_k):
    prod = 1.0
    for m in range(len(alpha_i)):
        val = E_He3_normalized(alpha_i[m], alpha_j[m], alpha_k[m])
        if val == 0.0: return 0.0
        prod *= val
    return prod

def petsc_to_scipy(petsc_mat):
    """Convertit une matrice PETSc en scipy.sparse.csr_matrix"""
    indptr, indices, data = petsc_mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=petsc_mat.getSize())

def resoudre_ssfem_scipy(D_rff, ordrePC):
    dim = 2 * D_rff
    
    if not os.path.exists("support_sparse.npy"):
        print("Erreur: support_sparse.npy introuvable.")
        sys.exit(1)
        
    S = np.load("support_sparse.npy")
    P_reduit = len(S)
    print(f"-> Résolution SSFEM-Sparse via SciPy (D={D_rff}, P_reduit={P_reduit})")

    # 1. INITIALISATION MEF++
    petsc4py.init(sys.argv)
    mefpp.initialise("conduction_trou")
    mefpp.litEtExecuteActionsDansCollection()
    gfc = mefpp.reqCollectionDeCorps().reqCorps("conduction_trou").reqGFC()

    champ_wx = gfc.reqChamp("omega_x")
    champ_wy = gfc.reqChamp("omega_y")
    champ_phase = gfc.reqChamp("phase_rff")
    pp_reinterpole = gfc.reqPP("pp_interpoleK")
    pp_assemblage = gfc.reqPP("ppAssMatEtRes")
    
    matK_petsc = gfc.reqMatricePETSc("MatK").reqMat()
    residu_petsc = gfc.reqVecteurPETSc("Residu").reqVec()

    # CORRECTION : Réinitialisation de la température pour obtenir un F_ext pur
    T_imp = gfc.reqVecteurPETSc("T_imp").reqVec()
    T_imp.set(0.0)
    T_imp.assemble()
    gfc.reqPP("pp_visualisation_T").execute()

    # 2. EXTRACTION DES MATRICES K_i
    facteur_norm = 0.0401 * np.sqrt(1.0 / D_rff)
    rng = np.random.default_rng(42)
    w = rng.normal(0, 1.0/0.12, (D_rff, 2))
    
    matrices_K_scipy = []
    
    # Mode 0 (Moyenne)
    champ_wx.asgnValeur(0.0); champ_wy.asgnValeur(0.0); champ_phase.asgnValeur(-np.pi/2.0)
    pp_reinterpole.execute(); pp_assemblage.execute()
    K0_petsc = matK_petsc.duplicate(); matK_petsc.copy(result=K0_petsc)
    K0_sp = petsc_to_scipy(K0_petsc)
    matrices_K_scipy.append(K0_sp)
    
    # F_ext
    F_ext = np.array(residu_petsc.getArray())
    N_noeuds = len(F_ext)
    
    print("Extraction des blocs physiques depuis MEF++...")
    for j in range(D_rff):
        champ_wx.asgnValeur(float(w[j, 0])); champ_wy.asgnValeur(float(w[j, 1]))
        
        # COS
        champ_phase.asgnValeur(0.0)
        pp_reinterpole.execute(); pp_assemblage.execute()
        mat_cos = matK_petsc.duplicate(); matK_petsc.copy(result=mat_cos)
        mat_cos.axpy(-1.0, K0_petsc)
        mat_cos.scale(facteur_norm)
        matrices_K_scipy.append(petsc_to_scipy(mat_cos))
        
        # SIN
        champ_phase.asgnValeur(-np.pi/2.0)
        pp_reinterpole.execute(); pp_assemblage.execute()
        mat_sin = matK_petsc.duplicate(); matK_petsc.copy(result=mat_sin)
        mat_sin.axpy(-1.0, K0_petsc)
        mat_sin.scale(facteur_norm)
        matrices_K_scipy.append(petsc_to_scipy(mat_sin))

    mefpp.finalise()

    # 3. ASSEMBLAGE DE GALERKIN EN SCIPY PUR
    print("Assemblage de la matrice globale Galerkin en RAM...")
    K_global_blocks = [[None for _ in range(P_reduit)] for _ in range(P_reduit)]
    
    for j_idx in range(P_reduit):
        for k_idx in range(P_reduit):
            alpha_j = S[j_idx]; alpha_k = S[k_idx]
            bloc = sp.csr_matrix((N_noeuds, N_noeuds))
            
            for i in range(dim+1):
                alpha_i = np.zeros(dim, dtype=np.int16)
                if i > 0: alpha_i[i-1] = 1
                coeff = calcul_cijk_tuple(alpha_i, alpha_j, alpha_k)
                if abs(coeff) > 1e-12:
                    bloc = bloc + coeff * matrices_K_scipy[i]
            
            K_global_blocks[j_idx][k_idx] = bloc

    K_galerkin = sp.bmat(K_global_blocks, format='csr')
    
    F_galerkin = np.zeros(P_reduit * N_noeuds)
    idx_0 = next(i for i, alpha in enumerate(S) if np.sum(alpha) == 0)
    F_galerkin[idx_0 * N_noeuds : (idx_0 + 1) * N_noeuds] = F_ext

    # 4. RÉSOLUTION DIRECTE EXACTE
    print(f"Inversion directe du système ({K_galerkin.shape[0]} DDL) via SciPy (MUMPS/SuperLU)...")
    T_galerkin = spla.spsolve(K_galerkin, F_galerkin)
    
    # 5. EXTRACTION DE LA VARIANCE
    T0 = T_galerkin[idx_0 * N_noeuds : (idx_0 + 1) * N_noeuds]
    T_var = np.zeros(N_noeuds)
    
    for j_idx in range(P_reduit):
        if j_idx == idx_0: continue
        Tj = T_galerkin[j_idx * N_noeuds : (j_idx + 1) * N_noeuds]
        T_var += Tj**2
        
    # Vérification automatique
    if os.path.exists("mc_mean_100k.npy") and os.path.exists("mc_var_100k.npy"):
        MC_mean = np.load("mc_mean_100k.npy")
        MC_var = np.load("mc_var_100k.npy")
        err_m = np.linalg.norm(T0 - MC_mean) / np.linalg.norm(MC_mean)
        err_v = np.linalg.norm(T_var - MC_var) / np.linalg.norm(MC_var)
        print("\n[RÉSULTAT SSFEM-SPARSE VIA SCIPY]")
        print(f"Erreur L2 Moyenne  : {err_m*100:.2f} %")
        print(f"Erreur L2 Variance : {err_v*100:.2f} %")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        resoudre_ssfem_scipy(int(sys.argv[1]), int(sys.argv[2]))
    else:
        print("Usage: python ssfem_scipy.py 50 3")