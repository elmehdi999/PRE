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
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc

warnings.filterwarnings("ignore", category=RuntimeWarning)

class NISP_RFF:
    def __init__(self, D_rff, ordrePC, N_evaluations):
        self.D_rff = D_rff
        self.d = 2 * D_rff 
        self.p = ordrePC   
        self.N_evaluations = N_evaluations 
        
        self.P_total = math.comb(self.d + self.p, self.d)
        self.erreur_L2_moyenne = None
        self.erreur_L2_variance = None
        
        print(" Initialisation NISP-RFF\n")
        print(f"Ondes RFF (D)            : {self.D_rff}")
        print(f"Ordre du Chaos (P)       : {self.p}")
        print(f"Taille théorique (P_total)  : {self.P_total}")
        print(f"Evaluations cibles MEF++ (N): {self.N_evaluations}\n")
        
        petsc4py.init(sys.argv)

        prefixe = "carre" 
        mefpp.initialise(prefixe)
        mefpp.litEtExecuteActionsDansCollection()
        collection = mefpp.reqCollectionDeCorps()
        collection.lisDonneesDeBase([prefixe])
        corps = collection.reqCorps(prefixe)
        self.gfc = corps.reqGFC()

        self.vecteur_T_imp = self.gfc.reqVecteurPETSc("T_imp").reqVec()
        self.vec_K = self.gfc.reqVecteurPETSc("K_imp").reqVec()
        
        self.indices_K = np.arange(self.vec_K.getSize(), dtype=np.int32)
        self.indices_T = np.arange(self.vecteur_T_imp.getSize(), dtype=np.int32)
        
        self.matK = self.gfc.reqMatricePETSc("MatK").reqMat()
        self.residu = self.gfc.reqVecteurPETSc("Residu").reqVec()
        
        self.pp_assemblage = self.gfc.reqPP("ppAssMatEtRes")
        self.pp_import_K = self.gfc.reqPP("pp_import_K")
        self.pp_copie_T = self.gfc.reqPP("pp_copie_T_imp")

        self.pp_copie_T.execute()
        T_init = self.vecteur_T_imp.getValues(self.indices_T).copy()
        self.noeud_cible = int(np.argmax(T_init))
        print(f" Noeud d'intérêt ciblé sur le point chaud (Temp. initiale = {T_init[self.noeud_cible]:.2f}°C)")
        
        self.gfc.reqPP("pp_copie_X_elem").execute()
        self.gfc.reqPP("pp_copie_Y_elem").execute()
        
        vec_x_petsc = self.gfc.reqVecteurPETSc("Vec_X_elem").reqVec()
        vec_y_petsc = self.gfc.reqVecteurPETSc("Vec_Y_elem").reqVec()
        
        self.X_elem = vec_x_petsc.getValues(self.indices_K).copy()
        self.Y_elem = vec_y_petsc.getValues(self.indices_K).copy()
        print(f" Coordonnées extraites pour {len(self.X_elem)} éléments.")

    def initialiser_frequences(self, l_corr_override=None):
        print("\n Initialisation des fréquences RFF")
        l_corr = l_corr_override if l_corr_override is not None else 0.12
        print(f" Utilisation de l_corr = {l_corr}")
        
        rng = np.random.default_rng(42)
        self.w = rng.normal(0, 1.0/l_corr, (self.D_rff, 2))
        
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
        
        self.Y_eval = np.zeros(self.N_evaluations)
        self.Y_eval_full = np.zeros((self.N_evaluations, len(self.indices_T))) 
        self.T_mean_global = np.zeros(len(self.indices_T))
        
        fd_stdout = sys.stdout.fileno()
        fd_stderr = sys.stderr.fileno()
        old_stdout = os.dup(fd_stdout)
        old_stderr = os.dup(fd_stderr)
        devnull = os.open(os.devnull, os.O_WRONLY)
        
        start_time = time.time()
        
        for i in range(self.N_evaluations):
            xi_cos = self.Xi[i, 0::2]  
            xi_sin = self.Xi[i, 1::2]  
            K_total = 1.0 + self.facteur_norm * (xi_cos @ self.cos_phases + xi_sin @ self.sin_phases)
            
            K_total = np.clip(K_total, 0.05, None)
            
            self.vec_K.setValues(self.indices_K, K_total)
            self.vec_K.assemble()
            self.pp_import_K.execute()
            
            os.dup2(devnull, fd_stdout)
            os.dup2(devnull, fd_stderr)
            try:
                self.pp_assemblage.execute() 
                
                nom_sol = f"Sol_NISP_{i}"
                nom_res = f"Res_NISP_{i}"
                self.gfc.lireLigne(f'solveur_lin {nom_sol}(ProbDida) prefixe_options options_slin')
                self.gfc.lireLigne(f'pp_resolution_probleme {nom_res} [ProbDida,{nom_sol}(ProbDida)]')
                self.gfc.reqPP(nom_res).execute()
                
                self.pp_copie_T.execute()
            finally:
                os.dup2(old_stdout, fd_stdout)
                os.dup2(old_stderr, fd_stderr)
            
            T_courant = self.vecteur_T_imp.getValues(self.indices_T).copy()
            self.Y_eval[i] = T_courant[self.noeud_cible]
            self.Y_eval_full[i, :] = T_courant 
            self.T_mean_global += T_courant / self.N_evaluations
            
            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{self.N_evaluations}] Évaluations terminées (T_max = {self.Y_eval[i]:.2f}°C)")
                for f in glob.glob("resultats/T_resultat_ssfem*.*"):
                     try: os.remove(f)
                     except Exception: pass
        
        os.close(devnull)
        end_time = time.time()
        print(f" Temps d'évaluation MEF++ : {end_time - start_time:.2f} secondes.")
        np.save(f"vraie_solution_NISP_D{self.D_rff}.npy", self.T_mean_global)
                
    def generer_multi_indices(self, q=0.75):
        print(f"\n Construction de la base PC (troncature hyperbolique q={q}) ")
        indices = []
        for c in itertools.combinations(range(self.d + self.p), self.d):
            idx = tuple([c[0]] + [c[i] - c[i-1] - 1 for i in range(1, self.d)])
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
        print("\n Apprentissage sparse ")
        
        if getattr(self, 'noeud_cible', None) is None:
            self.noeud_cible = int(np.argmax(np.mean(self.Y_eval_full, axis=0)))
            
        if getattr(self, 'Y_eval', None) is None or np.max(self.Y_eval) == 0:
            self.Y_eval = self.Y_eval_full[:, self.noeud_cible]
            
        Y_target = self.Y_eval_full[:self.N_evaluations, :]
        
        print(" Recherche automatique du nombre optimal de termes (OMP-CV)...")
        # On limite le nombre d'atomes au strict minimum entre :
        # 1. Le nombre de polynômes candidats disponibles (Psi.shape[1])
        # 2. La taille des données d'entraînement dans un pli de cross-validation (80% de N)
        n_features = self.Psi.shape[1]
        n_samples_cv = int(0.8 * self.N_evaluations) # Pour cv=5
        max_termes = min(n_features, n_samples_cv)

        omp_cv = OrthogonalMatchingPursuitCV(cv=5, max_iter=max_termes, n_jobs=-1)
        omp_cv.fit(self.Psi, self.Y_eval[:self.N_evaluations])
        
        indices_actifs_cv = np.where(np.abs(omp_cv.coef_) > 1e-10)[0]
        n_opt = len(indices_actifs_cv)
        if n_opt == 0: n_opt = 1
        print(f" L'algorithme CV a retenu {n_opt} termes optimaux.")
        
        print(" OMP-Multi-Sorties globale ")
        omp_global = OrthogonalMatchingPursuit(n_nonzero_coefs=n_opt, fit_intercept=False)
        omp_global.fit(self.Psi, Y_target)
        self.coefficients = omp_global.coef_ 
        
        self.PCE_mean_field = self.coefficients[:, 0]
        self.PCE_var_field = np.sum(self.coefficients[:, 1:]**2, axis=1)
        
        t_moyen = self.PCE_mean_field[self.noeud_cible] 
        t_var = self.PCE_var_field[self.noeud_cible] 
        
        print("\n Résultats analytiques de NISP-RFF sur le noeud cible")
        print(f"Moyenne empirique LHS          : {np.mean(self.Y_eval[:self.N_evaluations]):.4f} °C")
        print(f"Espérance E[T] analytique      : {t_moyen:.4f} °C")
        print(f"Variance Var[T] analytique     : {t_var:.4f}")
        """
        if os.path.exists("mc_mean_10k.npy") and os.path.exists("mc_var_10k.npy"):
            MC_mean = np.load("mc_mean_10k.npy")
            MC_var = np.load("mc_var_10k.npy")
            ref_utilisee = "10k"
        elif os.path.exists("mc_mean_100k.npy") and os.path.exists("mc_var_100k.npy"):
            MC_mean = np.load("mc_mean_100k.npy")
            MC_var = np.load("mc_var_100k.npy")
            ref_utilisee = "100k"
        elif os.path.exists("vraie_solution_mc_spatial.npy") and os.path.exists("variance_mc_spatial.npy"):
            MC_mean = np.load("vraie_solution_mc_spatial.npy")
            MC_var = np.load("variance_mc_spatial.npy")
            ref_utilisee = "100k"
        else:
            MC_mean, MC_var, ref_utilisee = None, None, None
        """
        MC_mean = None
        MC_var = None
        ref_utilisee = None
        
        if MC_mean is not None:
            self.erreur_L2_moyenne = np.linalg.norm(self.PCE_mean_field - MC_mean) / np.linalg.norm(MC_mean)
            self.erreur_L2_variance = np.linalg.norm(self.PCE_var_field - MC_var) / np.linalg.norm(MC_var)
            
            print(f"\n Erreur spatiale globale L2 (Face à réf. {ref_utilisee})")
            print(f"Erreur L2 sur la Moyenne       : {self.erreur_L2_moyenne:.4e} ({self.erreur_L2_moyenne*100:.2f} %)")
            print(f"Erreur L2 sur la Variance      : {self.erreur_L2_variance:.4e} ({self.erreur_L2_variance*100:.2f} %)")
            print("="*60)

        print("\n OMP Mono-Nœud (Préparation de la base pour SSFEM-Sparse)")
        indices_actifs = indices_actifs_cv
        
        if 0 not in indices_actifs:
            indices_actifs = np.insert(indices_actifs, 0, 0)
        indices_actifs = np.sort(indices_actifs)
            
        active_multi_indices = np.array([self.multi_indices[idx] for idx in indices_actifs], dtype=np.int16)
        np.save("support_sparse.npy", active_multi_indices)
        
        print(f"\ Support sparse strict sauvegardé ({len(indices_actifs)} polynômes actifs).")

    def exporter_resultats(self):
        print("\n Exportation fichier .vtu")
        """
        if os.path.exists("mc_mean_10k.npy"):
            MC_mean = np.load("mc_mean_10k.npy")
        elif os.path.exists("vraie_solution_mc_spatial.npy"):
            MC_mean = np.load("vraie_solution_mc_spatial.npy")
        else:
            MC_mean = None
        """
        MC_mean = None
    
        if MC_mean is not None:
            vec_exact = self.gfc.reqVecteurPETSc("T_exacte_vec").reqVec()
            vec_exact.setValues(self.indices_T, MC_mean)
            vec_exact.assemble()
            self.gfc.reqPP("pp_visu_Texacte").execute()
            print(" Fichier Monte Carlo (Moyenne) chargé pour la comparaison.")
        else:
            print(" Attention: Fichier MC introuvable pour la comparaison.")

        residu_backup = self.residu.duplicate()
        self.residu.copy(result=residu_backup)
        
        self.residu.setValues(self.indices_T, self.PCE_mean_field)
        self.residu.assemble()
        
        self.gfc.lireLigne('pp_copie_vecteur_dans_champs pp_push_nisp [Residu, T_exporte, T]')
        self.gfc.reqPP("pp_push_nisp").execute()
        
        residu_backup.copy(result=self.residu)
        residu_backup.destroy()
        
        nom_fichier = f"resultats/Comparaison_NISP_MC_D{self.D_rff}_P{self.p}"
        self.gfc.lireLigne(f'pp_exportation exp_finale [T_ssfem, "{nom_fichier}",0,true,false,false,false]')
        self.gfc.reqPP("exp_finale").execute()
        
        print(f" Fichier '{nom_fichier}.vtu' généré.")

    def finalise(self):
        mefpp.finalise()

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        D = int(sys.argv[1])
        P = int(sys.argv[2])
        N = int(sys.argv[3])
        l_corr_val = float(sys.argv[4]) if len(sys.argv) == 5 else None

        nisp = NISP_RFF(D, P, N)
        nisp.initialiser_frequences(l_corr_val)
        nisp.generer_plan_experience()
        
        l_cache_str = f"_lcorr{l_corr_val}" if l_corr_val is not None else ""
        cache = f"Y_eval_full_D{D}_N{N}{l_cache_str}.npy"
        
        nisp.evaluer_boite_noire()
        np.save(cache, {'Y_eval_full': nisp.Y_eval_full, 'Xi': nisp.Xi})
            
        nisp.generer_multi_indices(q=0.75)  
        nisp.evaluer_polynomes_hermite()
        nisp.regression_omp()
        nisp.exporter_resultats()  
        nisp.finalise()
    else:
        print("Usage : python nisp_rff.py <D_rff> <Ordre_P> <N_evaluations> [l_corr_optionnel]")
        print("Exemple : python nisp_rff.py 30 3 1024 0.12")
