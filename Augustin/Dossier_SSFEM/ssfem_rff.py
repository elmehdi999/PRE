########################################################################################################
# Code SSFEM RFF (Version Hybride Sparse - PCE Driven Galerkin)
# Auteur: El Mehdi EN-NAHAS 
########################################################################################################

import numpy.random as rd
import numpy as np
import sys
import os
import math
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc
 
import ks_stat
import matplotlib.pyplot as plt

# --- FONCTIONS ANALYTIQUES (Base Orthonormée) ---
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
# --------------------------------------------------------

class ssfem:
    def __init__(self, D_rff, ordrePC):
        self.D_rff = D_rff
        self.ordreKL = 2 * D_rff  
        self.ordrePC = ordrePC
        
        self.P_total = math.comb(self.ordreKL + self.ordrePC, self.ordreKL)
        
        print(f"Dimension stochastique (d) : {self.ordreKL}")
        print(f"Ordre du Chaos (P)         : {self.ordrePC}")
        
        if not os.path.exists("support_sparse.npy"):
            print("ERREUR : support_sparse.npy introuvable. Exécutez nisp_rff.py d'abord.")
            sys.exit(1)
        self.S = np.load("support_sparse.npy")
        self.P_reduit = len(self.S)
        print(f"SSFEM SPARSE : Réduction aux {self.P_reduit} blocs actifs trouvés par NISP\n")

        prefixe = "conduction_trou"
        petsc4py.init(sys.argv)
        mefpp.initialise(prefixe)
        mefpp.litEtExecuteActionsDansCollection() 
        collection=mefpp.reqCollectionDeCorps()
        collection.lisDonneesDeBase([prefixe])
        corps=collection.reqCorps(prefixe)
        self.gfc = corps.reqGFC()

        self.matK = self.gfc.reqMatricePETSc("MatK").reqMat()
        self.matK.assemble()
        self.residu = self.gfc.reqVecteurPETSc("Residu").reqVec()
        self.residu.assemble()

        self.champ_wx = self.gfc.reqChamp("omega_x")
        self.champ_wy = self.gfc.reqChamp("omega_y")
        self.champ_phase = self.gfc.reqChamp("phase_rff")
        self.pp_reinterpole = self.gfc.reqPP("pp_interpoleK")
        self.pp_assemblageMatEtRes = self.gfc.reqPP("ppAssMatEtRes")

        self.l_matK = []

    def assemblage_premier(self):
        D = self.D_rff 
        l_corr = 0.12
        facteur_norm = 0.0401 * np.sqrt(1.0 / D)
        
        rng = np.random.default_rng(42)
        self.w = rng.normal(0, 1.0/l_corr, (D, 2))
        
        T_imp = self.gfc.reqVecteurPETSc("T_imp").reqVec()
        T_imp.set(0.0)
        T_imp.assemble()
        self.gfc.reqPP("pp_visualisation_T").execute() 
        
        # MATRICE 0 (Moyenne)
        self.champ_wx.asgnValeur(0.0)
        self.champ_wy.asgnValeur(0.0)
        self.champ_phase.asgnValeur(-np.pi / 2.0)
        self.pp_reinterpole.execute()
        self.pp_assemblageMatEtRes.execute()
        
        K0 = self.matK.duplicate()
        self.matK.copy(result=K0)
        self.l_matK.append(K0)
        
        self.vrai_F = self.residu.duplicate()
        self.residu.copy(result=self.vrai_F)

        if self.ordreKL == 0:
            return

        # --- PURGE DE DIRICHLET ---
        # Identification des DDL pénalisés (Dirichlet) dans K0
        diag_K0 = K0.getDiagonal().getArray()
        dofs_dirichlet = np.where(diag_K0 > 1e15)[0].astype(np.int32)
        is_dirichlet = PETSc.IS().createGeneral(dofs_dirichlet)
        print(f"Purge Dirichlet : {len(dofs_dirichlet)} DDL identifiés et nettoyés sur les matrices de fluctuation.")
        
        # Création d'une copie propre de K0 avec des zéros sur les DDL Dirichlet
        K0_clean = K0.duplicate()
        K0.copy(result=K0_clean)
        K0_clean.zeroRowsColumns(is_dirichlet, diag=0.0)
            
        for j in range(D):
            self.champ_wx.asgnValeur(float(self.w[j, 0]))
            self.champ_wy.asgnValeur(float(self.w[j, 1]))
            
            # MATRICE COS 
            self.champ_phase.asgnValeur(0.0)
            self.pp_reinterpole.execute()
            self.pp_assemblageMatEtRes.execute()
            
            mat_cos = self.matK.duplicate()
            self.matK.copy(result=mat_cos)
            mat_cos.zeroRowsColumns(is_dirichlet, diag=0.0) # Purge avant soustraction
            mat_cos.axpy(-1.0, K0_clean) # Soustraction propre
            mat_cos.scale(facteur_norm)
            self.l_matK.append(mat_cos)

            # MATRICE SIN 
            self.champ_phase.asgnValeur(-np.pi / 2.0)
            self.pp_reinterpole.execute()
            self.pp_assemblageMatEtRes.execute()
            
            mat_sin = self.matK.duplicate()
            self.matK.copy(result=mat_sin)
            mat_sin.zeroRowsColumns(is_dirichlet, diag=0.0) # Purge avant soustraction
            mat_sin.axpy(-1.0, K0_clean) # Soustraction propre
            mat_sin.scale(facteur_norm)
            self.l_matK.append(mat_sin)
              
    def assemblage_second(self):
        dim = self.ordreKL
        self.K_jk = [[None for _ in range(self.P_reduit)] for _ in range(self.P_reduit)]
        
        for j_idx in range(self.P_reduit):
            for k_idx in range(j_idx, self.P_reduit):
                alpha_j = self.S[j_idx]
                alpha_k = self.S[k_idx]
                
                self.K_jk[j_idx][k_idx] = self.matK.duplicate()
                self.K_jk[j_idx][k_idx].zeroEntries()
                
                for i in range(dim+1):
                    alpha_i = np.zeros(dim, dtype=np.int16)
                    if i > 0: alpha_i[i-1] = 1
                        
                    coeff = calcul_cijk_tuple(alpha_i, alpha_j, alpha_k)
                    
                    if abs(coeff) > 1e-12:
                        temp = self.l_matK[i].duplicate()
                        self.l_matK[i].copy(result=temp)
                        temp.scale(coeff)
                        self.K_jk[j_idx][k_idx].axpy(1.0, temp)

                self.K_jk[j_idx][k_idx].assemble()

                if j_idx != k_idx:
                    self.K_jk[k_idx][j_idx] = self.matK.duplicate()
                    self.K_jk[j_idx][k_idx].copy(result=self.K_jk[k_idx][j_idx])

        self.K_assemble = PETSc.Mat().createNest(self.K_jk)
        self.K_assemble.assemble()
        
        # --- CONVERSION AIJ ---
        print("Conversion de la matrice MatNest en format séquentiel (AIJ)...")
        self.K_aij = self.K_assemble.convert("aij")
        self.K_aij.assemble()

    def assemblage_F(self):
        N = self.vrai_F.getSize() 
        F_array = np.zeros(self.P_reduit * N)
        
        # On place F_ext uniquement dans le bloc correspondant au mode 0 (espérance)
        for j_idx in range(self.P_reduit):
            if np.sum(self.S[j_idx]) == 0:
                F_array[j_idx * N : (j_idx + 1) * N] = self.vrai_F.getArray()
        
        # Création directe du vecteur AIJ
        self.F_aij = PETSc.Vec().createSeq(self.P_reduit * N)
        self.F_aij.setArray(F_array)
        self.F_aij.assemble()
        
    def construction_T(self):
        N = self.matK.getSize()[0] 
        liste_construction = []

        for j in range(self.P_reduit):
            Tj = PETSc.Vec().createSeq(N)
            Tj.set(0.0)
            Tj.assemble()
            liste_construction.append(Tj)
        
        self.T_assemble = PETSc.Vec().createNest(liste_construction)
        self.T_assemble.assemble()

        # Vecteur de destination AIJ global
        self.T_aij = self.K_aij.createVecRight()
        self.T_aij.set(0.0)
        self.T_aij.assemble()

    def resolution_lineaire(self):
        ksp = PETSc.KSP().create()
        ksp.setOperators(self.K_aij)
        ksp.setType('preonly')
        
        pc = ksp.getPC()
        pc.setType('lu')
        try:
            pc.setFactorSolverType('mumps')
        except PETSc.Error:
            print("Information: Solveur MUMPS non trouvé, utilisation du solveur LU par défaut.")

        taille_globale = self.P_reduit * self.matK.getSize()[0]
        print(f"\nRésolution du système global ({taille_globale} DDL) par factorisation LU exacte...")
        
        try:
            ksp.solve(self.F_aij, self.T_aij)

            if ksp.is_converged:
                print("Résolution exacte terminée.")
                
                # Réaffectation des données globales dans les blocs séparés
                T_array = self.T_aij.getArray()
                N_noeuds = self.matK.getSize()[0]
                
                for j_idx in range(self.P_reduit):
                    T_sub = self.T_assemble.getNestSubVecs()[j_idx]
                    T_sub.setArray(T_array[j_idx * N_noeuds : (j_idx + 1) * N_noeuds])
                    T_sub.assemble()
            else:
                print("ERREUR : La résolution n'a pas convergé.")
                sys.exit(1)
                
            # Benchmark optionnel
            try:
                idx_0 = 0
                for i, alpha in enumerate(self.S):
                    if np.sum(alpha) == 0: idx_0 = i; break
                        
                T0 = self.T_assemble.getNestSubVecs()[idx_0]
                valeurs_T0 = np.array(T0.getArray())
                fichier_ref = "mc_mean_100k.npy"
                if not os.path.exists(fichier_ref):
                    fichier_ref = "vraie_solution_mc_spatial.npy"
                
                if os.path.exists(fichier_ref):
                    valeurs_exactes = np.load(fichier_ref)
                    if len(valeurs_exactes) == len(valeurs_T0):
                        erreur_max = float(np.max(np.abs(valeurs_T0 - valeurs_exactes)))
                        print(f"\n---> RESULTAT_BENCHMARK_ERREUR={erreur_max:.5f} <---\n")
            except Exception as e:
                pass
                
        except PETSc.Error as e:
            print(f"Erreur PETSc : {e}")
            sys.exit(1)
            
        return self.T_assemble

    def exporter_statistiques(self):
        print("Calcul de la Variance Analytique Globale (Exacte)...")

        idx_0 = 0
        for i, alpha in enumerate(self.S):
            if np.sum(alpha) == 0: idx_0 = i; break

        T0 = self.T_assemble.getNestSubVecs()[idx_0]
        self.T_imp = self.gfc.reqVecteurPETSc("T_imp").reqVec()
        T0.copy(result=self.T_imp)
        self.T_imp.assemble()
        
        T_var = T0.duplicate()
        T_var.set(0.0)
        temp = T0.duplicate()
        
        for j_idx in range(self.P_reduit):
            if j_idx == idx_0: continue
            
            Tj = self.T_assemble.getNestSubVecs()[j_idx]
            temp.pointwiseMult(Tj, Tj) 
            T_var.axpy(1.0, temp)
            
        T_var.assemble()
        
        # Validation diagnostique automatique
        mc_mean_file = "mc_mean_100k.npy"
        mc_var_file = "mc_var_100k.npy"
        if os.path.exists(mc_mean_file) and os.path.exists(mc_var_file):
            MC_mean = np.load(mc_mean_file)
            MC_var = np.load(mc_var_file)
            
            erreur_moyenne = np.linalg.norm(T0.getArray() - MC_mean) / np.linalg.norm(MC_mean)
            erreur_variance = np.linalg.norm(T_var.getArray() - MC_var) / np.linalg.norm(MC_var)
            
            print(f"\n[RÉSULTAT SSFEM-SPARSE SUR TOUT LE MAILLAGE]")
            print(f"Erreur L2 Moyenne analytique  : {erreur_moyenne*100:.2f} %")
            print(f"Erreur L2 Variance analytique : {erreur_variance*100:.2f} %")
        else:
            print("Fichiers MC_100k introuvables. Diagnostique L2 ignoré.")

    def finalise(self):
        mefpp.finalise()

######################  
# Exécution
######################
if __name__ == "__main__":
    if len(sys.argv) == 3:
        a = ssfem(int(sys.argv[1]), int(sys.argv[2]))
        a.assemblage_premier()
        a.assemblage_second()
        a.assemblage_F()
        a.construction_T()
        a.resolution_lineaire()
        a.exporter_statistiques()
        a.finalise()
    else:
        print("Usage : python ssfem_rff.py <D_rff> <Ordre_P>")