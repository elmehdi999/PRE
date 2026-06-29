########################################################################################################
# Code NISP (Non-Intrusive Sparse Polynomial Chaos) - Conduction thermique 2D RFF
# Auteur: El Mehdi EN-NAHAS 
########################################################################################################

import numpy as np
import sys
import math
import itertools
import time
import os
import glob
from scipy.stats import qmc
from scipy.stats import norm
import numpy.polynomial.hermite_e as hermite_e 
from sklearn.linear_model import OrthogonalMatchingPursuit
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc

class NISP_RFF:
    def __init__(self, D_rff, ordrePC, N_evaluations):
        self.D_rff = D_rff
        self.d = 2 * D_rff 
        self.p = ordrePC   
        self.N_evaluations = N_evaluations 
        
        self.P_total = math.comb(self.d + self.p, self.d)
        
        print("="*60)
        print(" INITIALISATION NISP-RFF (RÉGRESSION SPARSE OMP)")
        print("="*60)
        print(f"Ondes RFF (D)            : {self.D_rff}")
        print(f"Variables stochastiques (d) : {self.d}")
        print(f"Ordre du Chaos (P)          : {self.p}")
        print(f"Taille théorique (P_total)  : {self.P_total}")
        print(f"Evaluations cibles MEF++ (N): {self.N_evaluations}")
        print("="*60)
        
        # Nettoyage console
        opts = PETSc.Options()
        for opt in ['options_slinksp_atol', 'options_slinksp_divtol', 'options_slinksp_max_it', 'options_slinksp_rtol']:
            if opts.hasName(opt):
                opts.delValue(opt)

        prefixe = "conduction_trou"
        petsc4py.init(sys.argv)
        mefpp.initialise(prefixe)
        mefpp.litEtExecuteActionsDansCollection()
        collection = mefpp.reqCollectionDeCorps()
        collection.lisDonneesDeBase([prefixe])
        corps = collection.reqCorps(prefixe)
        self.gfc = corps.reqGFC()

        self.vecteur_T_imp = self.gfc.reqVecteurPETSc("T_imp").reqVec()
        self.vec_K = self.gfc.reqVecteurPETSc("K_imp").reqVec()
        self.indices_K = np.arange(self.vec_K.getSize(), dtype=np.int32)
        
        self.matK = self.gfc.reqMatricePETSc("MatK").reqMat()
        self.residu = self.gfc.reqVecteurPETSc("Residu").reqVec()
        
        self.pp_assemblage = self.gfc.reqPP("ppAssMatEtRes")
        self.pp_import_K = self.gfc.reqPP("pp_import_K")
        
        # --- CORRECTION : Réintégration des opérateurs MEF++ manquants ---
        self.pp_resolution = self.gfc.reqPP("resolution")
        self.pp_copie_T = self.gfc.reqPP("pp_copie_T_imp")
        
        # --- EXTRACTION SÉCURISÉE DES COORDONNÉES SPATIALES ---
        self.gfc.reqPP("pp_copie_X_elem").execute()
        self.gfc.reqPP("pp_copie_Y_elem").execute()
        
        vec_x_petsc = self.gfc.reqVecteurPETSc("Vec_X_elem").reqVec()
        vec_y_petsc = self.gfc.reqVecteurPETSc("Vec_Y_elem").reqVec()
        
        self.X_elem = vec_x_petsc.getValues(self.indices_K)
        self.Y_elem = vec_y_petsc.getValues(self.indices_K)
        print(f"-> Coordonnées extraites pour {len(self.X_elem)} éléments.")

    def initialiser_frequences(self):
        print("\n--- Initialisation des fréquences RFF ---")
        l_corr = 0.12
        rng = np.random.default_rng(42)
        self.w = rng.normal(0, 1.0/l_corr, (self.D_rff, 2))
        self.facteur_norm = 0.3 * np.sqrt(1.0 / self.D_rff)
        print("Fréquences générées.")

    def generer_plan_experience(self):
        print(f"\n--- Génération du LHS ({self.N_evaluations} échantillons) ---")
        sampler = qmc.LatinHypercube(d=self.d, seed=42)
        sample_uniform = sampler.random(n=self.N_evaluations)
        self.Xi = norm.ppf(sample_uniform)
        
    def evaluer_boite_noire(self):
        print("\n--- Évaluation du modèle MEF++ (Boîte Noire) ---")
        N_noeuds = self.vecteur_T_imp.getSize()
        self.indices_T = np.arange(N_noeuds, dtype=np.int32)
        
        # 1. On trouve automatiquement le "point chaud" (bord du trou, ~19°C)
        # CORRECTION ICI : on utilise vec_K de manière robuste (comme dans la boucle)
        K_init = np.ones_like(self.X_elem)
        self.vec_K.setValues(self.indices_K, K_init)
        self.vec_K.assemble()
        self.pp_import_K.execute()
        
        self.pp_assemblage.execute()
        self.pp_resolution.execute()
        self.pp_copie_T.execute()
        T_ref = self.vecteur_T_imp.getValues(self.indices_T)
        self.noeud_cible = np.argmax(T_ref)
        print(f"-> Noeud d'intérêt ciblé sur le point chaud (Temp. initiale = {T_ref[self.noeud_cible]:.2f}°C)")
        
        self.Y_eval = np.zeros(self.N_evaluations)
        self.T_mean_global = np.zeros(N_noeuds)
        
        fd_stdout = sys.stdout.fileno()
        fd_stderr = sys.stderr.fileno()
        old_stdout = os.dup(fd_stdout)
        old_stderr = os.dup(fd_stderr)
        devnull = os.open(os.devnull, os.O_WRONLY)
        
        start_time = time.time()
        
        for i in range(self.N_evaluations):
            # Calcul Numpy du K_total (parfait et validé)
            K_total = np.ones_like(self.X_elem)
            
            for j in range(self.D_rff):
                phase = self.w[j, 0] * self.X_elem + self.w[j, 1] * self.Y_elem
                xi_cos = self.Xi[i, 2*j]
                xi_sin = self.Xi[i, 2*j + 1]
                K_total += self.facteur_norm * (xi_cos * np.cos(phase) + xi_sin * np.sin(phase))
            
            # Clip physique (Conductivité toujours strictement positive)
            K_total = np.clip(K_total, 0.05, None)
            
            # Injection
            self.vec_K.setValues(self.indices_K, K_total)
            self.vec_K.assemble()
            self.pp_import_K.execute()
            
            # Résolution native (100% MEF++)
            os.dup2(devnull, fd_stdout)
            os.dup2(devnull, fd_stderr)
            try:
                self.pp_assemblage.execute() 
                self.pp_resolution.execute()
                self.pp_copie_T.execute()
            finally:
                os.dup2(old_stdout, fd_stdout)
                os.dup2(old_stderr, fd_stderr)
            
            # Extraction des données
            T_courant = self.vecteur_T_imp.getValues(self.indices_T)
            self.Y_eval[i] = T_courant[self.noeud_cible]
            self.T_mean_global += T_courant / self.N_evaluations
            
            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{self.N_evaluations}] Évaluations terminées... (T_max = {self.Y_eval[i]:.2f}°C)")
                # Nettoyage intelligent (vtm, vtu, pvd générés dans la boucle)
                for f in glob.glob("resultats/T_resultat_ssfem*.*"):
                     try: os.remove(f)
                     except Exception: pass
        
        os.close(devnull)
        end_time = time.time()
        print(f"-> Temps d'évaluation Boîte Noire : {end_time - start_time:.2f} secondes.")
        np.save(f"vraie_solution_NISP_D{self.D_rff}.npy", self.T_mean_global)
                
    def generer_multi_indices(self):
        print("\n--- Construction de la base du Chaos Polynomial ---")
        indices = []
        for c in itertools.combinations(range(self.d + self.p), self.d):
            idx = [c[0]] + [c[i] - c[i-1] - 1 for i in range(1, self.d)]
            indices.append(tuple(idx))
        self.multi_indices = indices
        print(f"-> Base générée : {len(self.multi_indices)} polynômes candidats.")
        
    def evaluer_polynomes_hermite(self):
        self.Psi = np.ones((self.N_evaluations, len(self.multi_indices)))
        for p_idx, alpha in enumerate(self.multi_indices):
            for j in range(self.d):
                if alpha[j] > 0:
                    coef = np.zeros(alpha[j] + 1)
                    coef[alpha[j]] = 1.0
                    val_hermite = hermite_e.hermeval(self.Xi[:, j], coef)
                    norm_factor = np.sqrt(math.factorial(alpha[j]))
                    self.Psi[:, p_idx] *= (val_hermite / norm_factor)
                    
    def regression_omp(self):
        print("\n--- Apprentissage Sparse (OMP) ---")
        n_coefs = min(int(0.1 * self.P_total), self.N_evaluations // 2)
        
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_coefs, fit_intercept=False)
        omp.fit(self.Psi, self.Y_eval)
        self.coefficients = omp.coef_
        
        t_moyen = self.coefficients[0] 
        t_var = np.sum(self.coefficients[1:]**2) 
        
        print("\n" + "="*60)
        print(" RÉSULTATS DU NISP-RFF SUR LE NOEUD CIBLE (Point Chaud)")
        print("="*60)
        print(f"Moyenne empirique évaluée      : {np.mean(self.Y_eval):.4f} °C")
        print(f"Espérance E[T] prédite (OMP)   : {t_moyen:.4f} °C")
        print(f"Variance Var[T] prédite (OMP)  : {t_var:.4f}")
        print(f"Polynômes non-nuls (actifs)    : {np.count_nonzero(self.coefficients)} / {self.P_total}")
        print("="*60)

    def exporter_resultats(self):
        print("\n--- Exportation VTU pour ParaView ---")
        
        # Astuce absolue anti-lock : On écrit la moyenne dans "T_exacte_vec" qui est totalement libre
        vec_exact = self.gfc.reqVecteurPETSc("T_exacte_vec").reqVec()
        vec_exact.setValues(self.indices_T, self.T_mean_global)
        vec_exact.assemble()
        self.gfc.reqPP("pp_visualisation_T").execute()
        self.gfc.reqPP("pp_visu_Texacte").execute()

        nom_fichier = f"resultats/Moyenne_NISP_D{self.D_rff}_P{self.p}"
        self.gfc.lireLigne(f'pp_exportation exp_finale [T_ssfem, "{nom_fichier}",0,true,false,false,false]')
        self.gfc.reqPP("exp_finale").execute()
        
        print(f"-> Succès : Fichier '{nom_fichier}.vtu' généré !")
        print("-> Pour visualiser la moyenne NISP dans ParaView, sélectionnez le champ 'T_exacte_scallin'.")

    def finalise(self):
        mefpp.finalise()

if __name__ == "__main__":
    if len(sys.argv) == 4:
        nisp = NISP_RFF(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        nisp.initialiser_frequences()
        nisp.generer_plan_experience()
        nisp.evaluer_boite_noire() 
        nisp.generer_multi_indices()
        nisp.evaluer_polynomes_hermite()
        nisp.regression_omp()
        nisp.exporter_resultats()  
        nisp.finalise()
    else:
        print("Usage : python nisp_rff.py <D_rff> <Ordre_P> <N_evaluations>")
        print("Exemple : python nisp_rff.py 100 3 200")