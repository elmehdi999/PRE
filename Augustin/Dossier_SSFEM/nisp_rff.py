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
import warnings
from scipy.stats import qmc
from scipy.stats import norm
import numpy.polynomial.hermite_e as hermite_e 
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc

warnings.filterwarnings("ignore", category=RuntimeWarning)

class NISP_RFF:
    def __init__(self, D_rff, ordrePC, N_evaluations):
        self.D_rff = D_rff
        self.d = 2 * D_rff 
        self.p = ordrePC   
        self.N_evaluations = N_evaluations #
        
        self.P_total = math.comb(self.d + self.p, self.d)
        self.erreur_L2_moyenne = None
        self.erreur_L2_variance = None
        
        print(" INITIALISATION NISP-RFF (RÉGRESSION SPARSE OMP-CV)\n")
        print(f"Ondes RFF (D)            : {self.D_rff}")
        print(f"Ordre du Chaos (P)          : {self.p}")
        print(f"Taille théorique (P_total)  : {self.P_total}")
        print(f"Evaluations cibles MEF++ (N): {self.N_evaluations}\n")
        
        # nettoyage console
        opts = PETSc.Options()
        for opt in ['options_slinksp_atol', 'options_slinksp_divtol', 'options_slinksp_max_it', 'options_slinksp_rtol']:
            if opts.hasName(opt):
                opts.delValue(opt)

        prefixe = "conduction_trou" # a remplacer pour chaque probleme
        petsc4py.init(sys.argv)
        mefpp.initialise(prefixe)
        mefpp.litEtExecuteActionsDansCollection()
        collection = mefpp.reqCollectionDeCorps()
        collection.lisDonneesDeBase([prefixe])
        corps = collection.reqCorps(prefixe)
        self.gfc = corps.reqGFC()

        self.vecteur_T_imp = self.gfc.reqVecteurPETSc("T_imp").reqVec()
        self.vec_K = self.gfc.reqVecteurPETSc("K_imp").reqVec()
        
        # definition des index des l'initialisation
        self.indices_K = np.arange(self.vec_K.getSize(), dtype=np.int32)
        self.indices_T = np.arange(self.vecteur_T_imp.getSize(), dtype=np.int32)
        
        self.matK = self.gfc.reqMatricePETSc("MatK").reqMat()
        self.residu = self.gfc.reqVecteurPETSc("Residu").reqVec()
        
        self.pp_assemblage = self.gfc.reqPP("ppAssMatEtRes")
        self.pp_import_K = self.gfc.reqPP("pp_import_K")
        self.pp_resolution = self.gfc.reqPP("resolution")
        self.pp_copie_T = self.gfc.reqPP("pp_copie_T_imp")
        
        # extraction des coordonnees spatiales
        self.gfc.reqPP("pp_copie_X_elem").execute()
        self.gfc.reqPP("pp_copie_Y_elem").execute()
        
        vec_x_petsc = self.gfc.reqVecteurPETSc("Vec_X_elem").reqVec()
        vec_y_petsc = self.gfc.reqVecteurPETSc("Vec_Y_elem").reqVec()
        
        self.X_elem = vec_x_petsc.getValues(self.indices_K)
        self.Y_elem = vec_y_petsc.getValues(self.indices_K)
        print(f" Coordonnées extraites pour {len(self.X_elem)} éléments.")

    def initialiser_frequences(self):
        print("\n Initialisation des fréquences RFF")
        l_corr = 0.12
        rng = np.random.default_rng(42)
        self.w = rng.normal(0, 1.0/l_corr, (self.D_rff, 2))
        
        # alignement physique de la variance (0.0401) pour correspondre au MC
        self.facteur_norm = 0.0401 * np.sqrt(1.0 / self.D_rff)
        
        phases = self.w[:, [0]] * self.X_elem + self.w[:, [1]] * self.Y_elem
        self.cos_phases = np.cos(phases)
        self.sin_phases = np.sin(phases)
        
        print("Fréquences générées.")

    def generer_plan_experience(self):
        print(f"\n Génération du LHS ({self.N_evaluations} échantillons) ---")
        sampler = qmc.LatinHypercube(d=self.d, seed=42)
        sample_uniform = sampler.random(n=self.N_evaluations)
        self.Xi = norm.ppf(sample_uniform)
        
    def evaluer_boite_noire(self):
        print("\n Evaluation MEF++")
        
        # 1. on trouve automatiquement le "point chaud" (bord du trou pour le probleme de plaque trouee)
        K_init = np.ones_like(self.X_elem)
        self.vec_K.setValues(self.indices_K, K_init)
        self.vec_K.assemble()
        self.pp_import_K.execute()
        
        self.pp_assemblage.execute()
        self.pp_resolution.execute()
        self.pp_copie_T.execute()
        T_ref = self.vecteur_T_imp.getValues(self.indices_T)
        self.noeud_cible = np.argmax(T_ref)
        print(f" Noeud d'intérêt ciblé sur le point chaud (Temp. initiale = {T_ref[self.noeud_cible]:.2f}°C)")
        
        self.Y_eval = np.zeros(self.N_evaluations)
        self.Y_eval_full = np.zeros((self.N_evaluations, len(self.indices_T))) # sauvegarde du champ complet
        self.T_mean_global = np.zeros(len(self.indices_T))
        
        fd_stdout = sys.stdout.fileno()
        fd_stderr = sys.stderr.fileno()
        old_stdout = os.dup(fd_stdout)
        old_stderr = os.dup(fd_stderr)
        devnull = os.open(os.devnull, os.O_WRONLY)
        
        start_time = time.time()
        
        for i in range(self.N_evaluations):
            # calcul numpy de K_total
            xi_cos = self.Xi[i, 0::2]  
            xi_sin = self.Xi[i, 1::2]  
            K_total = 1.0 + self.facteur_norm * (xi_cos @ self.cos_phases + xi_sin @ self.sin_phases)
            
            # clip physique 
            K_total = np.clip(K_total, 0.05, None)
            
            # injection
            self.vec_K.setValues(self.indices_K, K_total)
            self.vec_K.assemble()
            self.pp_import_K.execute()
            
            # resolution avec MEF++
            os.dup2(devnull, fd_stdout)
            os.dup2(devnull, fd_stderr)
            try:
                self.pp_assemblage.execute() 
                self.pp_resolution.execute()
                self.pp_copie_T.execute()
            finally:
                os.dup2(old_stdout, fd_stdout)
                os.dup2(old_stderr, fd_stderr)
            
            # extraction des donnees
            T_courant = self.vecteur_T_imp.getValues(self.indices_T)
            self.Y_eval[i] = T_courant[self.noeud_cible]
            self.Y_eval_full[i, :] = T_courant # sauvegarde du champ complet
            self.T_mean_global += T_courant / self.N_evaluations
            
            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{self.N_evaluations}] Évaluations terminées... (T_max = {self.Y_eval[i]:.2f}°C)")
                # nettoyage automatique des fichiers residus
                for f in glob.glob("resultats/T_resultat_ssfem*.*"):
                     try: os.remove(f)
                     except Exception: pass
        
        os.close(devnull)
        end_time = time.time()
        print(f" Temps d'évaluation MEF++ : {end_time - start_time:.2f} secondes.")
        np.save(f"vraie_solution_NISP_D{self.D_rff}.npy", self.T_mean_global)
                
    def generer_multi_indices(self, q=0.75):
        print(f"\n Construction de la base PC (troncature hyperbolique q={q}) ---")
        indices = []
        for c in itertools.combinations(range(self.d + self.p), self.d):
            idx = tuple([c[0]] + [c[i] - c[i-1] - 1 for i in range(1, self.d)])
            
            # troncature hyperbolique : norme-q <= p
            norme_q = sum(a**q for a in idx) ** (1.0/q) if sum(idx) > 0 else 0.0
            if norme_q <= self.p:
                indices.append(idx)
                
        self.multi_indices = indices
        self.P_effectif = len(indices)
        
        print(f" Base complète : {self.P_total} polynômes")
        print(f" Après troncature hyperbolique (q={q}) : {self.P_effectif} polynômes candidats")
        
    def evaluer_polynomes_hermite(self):
        self.Psi = np.ones((self.N_evaluations, self.P_effectif))
        for p_idx, alpha in enumerate(self.multi_indices):
            for j in range(self.d):
                if alpha[j] > 0:
                    coef = np.zeros(alpha[j] + 1)
                    coef[alpha[j]] = 1.0
                    val_hermite = hermite_e.hermeval(self.Xi[:, j], coef)
                    norm_factor = np.sqrt(math.factorial(alpha[j]))
                    self.Psi[:, p_idx] *= (val_hermite / norm_factor)
                    
    def regression_omp(self):
        print("\n Apprentissage PCE régularisé (RidgeCV global pour tout le maillage) ---")
        from sklearn.linear_model import RidgeCV
        
        # 1. Ajustement OMP-CV uniquement sur le point chaud pour l'analyse locale (facultatif/diagnostic)
        omp = OrthogonalMatchingPursuitCV(fit_intercept=False, cv=5)
        omp.fit(self.Psi, self.Y_eval)
        self.coefficients = omp.coef_
        
        # 2. RÉSOLUTION DU VERROU SPATIAL : Régression RidgeCV sur tout le maillage d'un coup
        # On teste plusieurs paramètres de pénalisation (alpha) par validation croisée
        alphas = np.logspace(-6, 3, 10)
        ridge = RidgeCV(alphas=alphas, fit_intercept=False, cv=5)
        ridge.fit(self.Psi, self.Y_eval_full) # taille : (N_evaluations, N_noeuds)
        
        # Les coefficients optimaux pour chaque nœud (P_effectif, N_noeuds)
        self.C_full = ridge.coef_.T 
        
        # 3. EXTRACTION ANALYTIQUE DES MOMENTS DU PCE (Sans aucun bypass empirique !)
        self.PCE_mean_field = self.C_full[0, :]                 # Coefficient de la constante c_0
        self.PCE_var_field = np.sum(self.C_full[1:, :]**2, axis=0) # Somme des c_j^2 pour j >= 1
        
        # Stats locales au point chaud pour l'affichage
        t_moyen = self.PCE_mean_field[self.noeud_cible]
        t_var = self.PCE_var_field[self.noeud_cible]
        
        print("\n")
        print(" RÉSULTATS DU NISP-RFF SUR LE NOEUD CIBLE (Point Chaud)")
        print(f"Moyenne empirique évaluée      : {np.mean(self.Y_eval):.4f} °C")
        print(f"Espérance E[T] analytique PCE  : {t_moyen:.4f} °C")
        print(f"Variance Var[T] analytique PCE : {t_var:.4f}")
        
        # calcul de l'erreur L2 globale vs référence
        fichier_ref_moy = "vraie_solution_mc_spatial.npy"
        fichier_ref_var = "variance_mc_spatial.npy"
        
        if os.path.exists(fichier_ref_moy) and os.path.exists(fichier_ref_var):
            MC_mean = np.load(fichier_ref_moy)
            MC_var = np.load(fichier_ref_var)
            
            self.erreur_L2_moyenne = np.linalg.norm(self.PCE_mean_field - MC_mean) / np.linalg.norm(MC_mean)
            self.erreur_L2_variance = np.linalg.norm(self.PCE_var_field - MC_var) / np.linalg.norm(MC_var)
            
            print("\n ERREUR SPATIALE GLOBALE L2 (Sur tout le maillage)")
            print(f"Erreur L2 sur la Moyenne       : {self.erreur_L2_moyenne:.4e} ({self.erreur_L2_moyenne*100:.2f} %)")
            print(f"Erreur L2 sur la Variance      : {self.erreur_L2_variance:.4e} ({self.erreur_L2_variance*100:.2f} %)")
        print("="*60)

    def exporter_resultats(self):
        print("\n Exportation VTU")
        
        # 1. pousser la moyenne MC dans le champ "T_exacte_scallin"
        fichier_ref_moy = "vraie_solution_mc_spatial.npy"
        if os.path.exists(fichier_ref_moy):
            MC_mean = np.load(fichier_ref_moy)
            vec_exact = self.gfc.reqVecteurPETSc("T_exacte_vec").reqVec()
            vec_exact.setValues(self.indices_T, MC_mean)
            vec_exact.assemble()
            self.gfc.reqPP("pp_visu_Texacte").execute()
            print(" Fichier Monte Carlo (Moyenne) chargé pour la comparaison.")
        else:
            print(" Attention: Fichier 'vraie_solution_mc_spatial.npy' introuvable.")

        # 2. pousser la moyenne NISP dans le champ "T_exporte"
        residu_backup = self.residu.duplicate()
        self.residu.copy(result=residu_backup)
        
        # injection securisee de la température moyenne NISP dans Residu
        self.residu.setValues(self.indices_T, self.PCE_mean_field)
        self.residu.assemble()
        
        # creation a la volee d'une regle de copie entre Residu et le champ T_exporte
        self.gfc.lireLigne('pp_copie_vecteur_dans_champs pp_push_nisp [Residu, T_exporte, T]')
        self.gfc.reqPP("pp_push_nisp").execute()
        
        residu_backup.copy(result=self.residu)
        residu_backup.destroy()
        
        # 3. exporter le fichier contenant les deux champs de température
        nom_fichier = f"resultats/Comparaison_NISP_MC_D{self.D_rff}_P{self.p}"
        self.gfc.lireLigne(f'pp_exportation exp_finale [T_ssfem, "{nom_fichier}",0,true,false,false,false]')
        self.gfc.reqPP("exp_finale").execute()
        
        print(f" Fichier '{nom_fichier}.vtu' généré.")
        print("  'T_exporte'        = Température Moyenne prédite par NISP-RFF")
        print("  'T_exacte_scallin' = Température Moyenne de référence Monte Carlo")

    def finalise(self):
        mefpp.finalise()

if __name__ == "__main__":
    if len(sys.argv) == 4:
        nisp = NISP_RFF(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        nisp.initialiser_frequences()
        nisp.generer_plan_experience()
        nisp.evaluer_boite_noire() 
        nisp.generer_multi_indices(q=0.75)  # troncature hyperbolique
        nisp.evaluer_polynomes_hermite()
        nisp.regression_omp()
        nisp.exporter_resultats()  
        nisp.finalise()
    else:
        print("Usage : python nisp_rff.py <D_rff> <Ordre_P> <N_evaluations>")
        print("Exemple : python nisp_rff.py 20 3 500")