import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
import os
import math
import petsc4py
from mefpp4py import mefpp

def calcul_cijk_tuple(alpha_i, alpha_j, alpha_k):
    def E_He3_normalized(a, b, c):
        if (a + b + c) % 2 != 0: return 0.0
        s = (a + b + c) // 2
        if s < a or s < b or s < c: return 0.0
        num = math.sqrt(math.factorial(a) * math.factorial(b) * math.factorial(c))
        den = math.factorial(s - a) * math.factorial(s - b) * math.factorial(s - c)
        return num / den

    prod = 1.0
    for m in range(len(alpha_i)):
        val = E_He3_normalized(alpha_i[m], alpha_j[m], alpha_k[m])
        if val == 0.0: return 0.0
        prod *= val
    return prod

def petsc_to_scipy(petsc_mat):
    petsc_mat.assemble()
    indptr, indices, data = petsc_mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=petsc_mat.getSize())

def resoudre_ssfem_scipy(D_rff, ordrePC):
    dim = 2 * D_rff
    
    if not os.path.exists("support_sparse.npy"):
        print("Erreur: support_sparse.npy introuvable.")
        sys.exit(1)
        
    S = np.load("support_sparse.npy")
    
    # SÉCURITÉ : Vérification de la cohérence dimensionnelle
    if len(S) > 0 and len(S[0]) != dim:
        print(f"\n[ERREUR DIMENSIONNELLE]")
        print(f"Vous avez lancé SSFEM avec D={D_rff} (d={dim}).")
        print(f"Mais 'support_sparse.npy' contient des multi-indices de taille d={len(S[0])}.")
        print(f"-> Veuillez d'abord exécuter : python nisp_rff.py {D_rff} {ordrePC} 1024")
        sys.exit(1)
        
    P_reduit = len(S)
    print(f"-> Résolution SSFEM-Sparse via SciPy (D={D_rff}, P_reduit={P_reduit})")

    # Initialisation MEF++
    petsc4py.init(sys.argv)
    mefpp.initialise("conduction_trou")
    mefpp.litEtExecuteActionsDansCollection()
    
    opts = petsc4py.PETSc.Options()
    for opt in ['options_slinksp_atol', 'options_slinksp_divtol', 'options_slinksp_max_it', 'options_slinksp_rtol']:
        if opts.hasName(opt): opts.delValue(opt)

    gfc = mefpp.reqCollectionDeCorps().reqCorps("conduction_trou").reqGFC()
    vec_K = gfc.reqVecteurPETSc("K_imp").reqVec()
    vec_T = gfc.reqVecteurPETSc("T_imp").reqVec()
    indices_K = np.arange(vec_K.getSize(), dtype=np.int32)
    indices_T = np.arange(vec_T.getSize(), dtype=np.int32)

    # 1. EXTRACTION DE L'ÉTAT DE RÉFÉRENCE PUR (K = 1.0)
    print("\nExtraction de la matrice de base K0 (K=1.0)...")
    vec_K.setValues(indices_K, np.ones(len(indices_K)))
    vec_K.assemble()
    gfc.reqPP("pp_import_K").execute()
    gfc.reqPP("ppAssMatEtRes").execute()
    
    # Le Fix : on utilise un solveur dynamique comme dans NISP pour avoir la vraie physique
    gfc.lireLigne('solveur_lin Sol_Base(ProbDida) prefixe_options options_slin')
    gfc.lireLigne('pp_resolution_probleme Reso_Base [ProbDida,Sol_Base(ProbDida)]')
    gfc.reqPP("Reso_Base").execute()
    gfc.reqPP("pp_copie_T_imp").execute()
    
    K0_sp = petsc_to_scipy(gfc.reqMatricePETSc("MatK").reqMat()) 
    T_true = vec_T.getValues(indices_T).copy() 
    N_noeuds = K0_sp.shape[0]

    print(f"-> Max T_ref MEF++ (K=1.0) : {np.max(T_true):.4f} °C")

    # Construction du vrai second membre purifié (F = K0 * T)
    F_ext = K0_sp.dot(T_true)

    # 2. EXTRACTION DES BLOCS DE FLUCTUATIONS
    print("Extraction des blocs physiques (purification algébrique locale)...")
    gfc.reqPP("pp_copie_X_elem").execute()
    gfc.reqPP("pp_copie_Y_elem").execute()
    X_elem = gfc.reqVecteurPETSc("Vec_X_elem").reqVec().getValues(indices_K)
    Y_elem = gfc.reqVecteurPETSc("Vec_Y_elem").reqVec().getValues(indices_K)

    facteur_norm = 0.0401 * np.sqrt(1.0 / D_rff)
    rng = np.random.default_rng(42)
    w = rng.normal(0, 1.0/0.12, (D_rff, 2))
    
    matrices_K_fluct = []
    
    # Remplacement de stdout pour cacher le spam de MEF++
    fd_stdout = sys.stdout.fileno()
    old_stdout = os.dup(fd_stdout)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd_stdout)
    
    try:
        for j in range(D_rff):
            # Onde Cosinus
            fluct_cos = facteur_norm * np.cos(w[j,0]*X_elem + w[j,1]*Y_elem)
            vec_K.setValues(indices_K, 1.0 + fluct_cos)
            vec_K.assemble()
            gfc.reqPP("pp_import_K").execute()
            gfc.reqPP("ppAssMatEtRes").execute()
            mat_cos = petsc_to_scipy(gfc.reqMatricePETSc("MatK").reqMat())
            matrices_K_fluct.append(mat_cos - K0_sp) # Soustraction avec le vrai K0
            
            # Onde Sinus
            fluct_sin = facteur_norm * np.sin(w[j,0]*X_elem + w[j,1]*Y_elem)
            vec_K.setValues(indices_K, 1.0 + fluct_sin)
            vec_K.assemble()
            gfc.reqPP("pp_import_K").execute()
            gfc.reqPP("ppAssMatEtRes").execute()
            mat_sin = petsc_to_scipy(gfc.reqMatricePETSc("MatK").reqMat())
            matrices_K_fluct.append(mat_sin - K0_sp) # Soustraction avec le vrai K0
    finally:
        os.dup2(old_stdout, fd_stdout)
        os.close(devnull)

    mefpp.finalise()

    # 3. ASSEMBLAGE GALERKIN SPATIAL EN PYTHON
    print("Assemblage de la matrice globale Galerkin en RAM...")
    K_global_blocks = [[None for _ in range(P_reduit)] for _ in range(P_reduit)]
    
    for j_idx in range(P_reduit):
        for k_idx in range(j_idx, P_reduit):
            alpha_j = S[j_idx]; alpha_k = S[k_idx]
            bloc = sp.csr_matrix((N_noeuds, N_noeuds))
            
            coeff_0 = calcul_cijk_tuple(np.zeros(dim, dtype=np.int16), alpha_j, alpha_k)
            if abs(coeff_0) > 1e-12:
                bloc = bloc + coeff_0 * K0_sp
                
            for i in range(1, dim+1):
                alpha_i = np.zeros(dim, dtype=np.int16)
                alpha_i[i-1] = 1
                coeff = calcul_cijk_tuple(alpha_i, alpha_j, alpha_k)
                if abs(coeff) > 1e-12:
                    bloc = bloc + coeff * matrices_K_fluct[i-1]
            
            K_global_blocks[j_idx][k_idx] = bloc
            if j_idx != k_idx:
                K_global_blocks[k_idx][j_idx] = bloc

    K_galerkin = sp.bmat(K_global_blocks, format='csr')
    
    F_galerkin = np.zeros(P_reduit * N_noeuds)
    idx_0 = next(i for i, alpha in enumerate(S) if np.sum(alpha) == 0)
    F_galerkin[idx_0 * N_noeuds : (idx_0 + 1) * N_noeuds] = F_ext

    # 4. RESOLUTION
    taille_globale = K_galerkin.shape[0]
    print(f"Inversion directe du système ({taille_globale} DDL) via SciPy (MUMPS/SuperLU)...")
    T_galerkin = spla.spsolve(K_galerkin, F_galerkin)
    
    # 5. POST-TRAITEMENT ET AFFICHAGE ABSOLU
    T0 = T_galerkin[idx_0 * N_noeuds : (idx_0 + 1) * N_noeuds]
    T_var = np.zeros(N_noeuds)
    for j_idx in range(P_reduit):
        if j_idx == idx_0: continue
        Tj = T_galerkin[j_idx * N_noeuds : (j_idx + 1) * N_noeuds]
        T_var += Tj**2
        
    print("\n[RÉSULTAT SSFEM-SPARSE VIA SCIPY]")
    print(f"Moyenne analytique spatiale (max) : {np.max(T0):.4f} °C")
    print(f"Variance analytique spatiale (max) : {np.max(T_var):.4f}")

    # COMPARAISON AVEC CACHE
    if os.path.exists("mc_mean_10k.npy") and os.path.exists("mc_var_10k.npy"):
        MC_mean = np.load("mc_mean_10k.npy")
        MC_var = np.load("mc_var_10k.npy")
        ref_name = "10k (Pur GMRES)"
    elif os.path.exists("mc_mean_100k.npy") and os.path.exists("mc_var_100k.npy"):
        MC_mean = np.load("mc_mean_100k.npy")
        MC_var = np.load("mc_var_100k.npy")
        ref_name = "100k (Pollué)"
    else:
        MC_mean = None

    if MC_mean is not None:
        err_m = np.linalg.norm(T0 - MC_mean) / np.linalg.norm(MC_mean)
        err_v = np.linalg.norm(T_var - MC_var) / np.linalg.norm(MC_var)
        print(f"\n[COMPARAISON SPATIALE L2 (Face au cache {ref_name})]")
        print(f"Max T_ref Monte Carlo (Sauvegarde) : {np.max(MC_mean):.4f} °C")
        print(f"Erreur L2 Moyenne  : {err_m*100:.2f} %")
        print(f"Erreur L2 Variance : {err_v*100:.2f} %")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        resoudre_ssfem_scipy(int(sys.argv[1]), int(sys.argv[2]))
    else:
        print("Usage: python ssfem_scipy.py <D> <P>")
