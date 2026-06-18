########################################################################################################
# Code SSFEM conduction thermique 2D - Version Benchmark Spatial Corrigée
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
 
import p_chaos 
import ks_stat
import matplotlib.pyplot as plt

class ssfem:
    def __init__(self, D_rff, ordrePC):
        self.D_rff = D_rff
        self.ordreKL = 2 * D_rff  # d = 2D (cos + sin)
        self.ordrePC = ordrePC
        
        # Calcul de la vraie taille de la base P_total (et non ordrePC)
        self.P_total = math.comb(self.ordreKL + self.ordrePC, self.ordreKL)
        
        print("\n==============================================")
        print(f"Dimension stochastique (d) : {self.ordreKL}")
        print(f"Ordre du Chaos (P)         : {self.ordrePC}")
        print(f"Taille de la base (P_total): {self.P_total}")
        print("==============================================\n")
        
        if self.P_total > 500:
            print("ATTENTION CRITIQUE : P_total est gigantesque !")
            print("La matrice bloc va exploser la memoire du cluster.")
            print("Pour tester SSFEM, vous DEVEZ reduire D_rff (ex: 2, 3 ou 4).\n")
        
        # 1. Initialisation
        prefixe = "conduction_trou"
        petsc4py.init(sys.argv)
        mefpp.initialise(prefixe)
        mefpp.litEtExecuteActionsDansCollection()
        collection=mefpp.reqCollectionDeCorps()
        collection.lisDonneesDeBase([prefixe])
        corps=collection.reqCorps(prefixe)
        self.gfc = corps.reqGFC()

        # 2. PRISE EN CHARGE RFF
        self.champ_wx = self.gfc.reqChamp("omega_x")
        self.champ_wy = self.gfc.reqChamp("omega_y")
        self.champ_phase = self.gfc.reqChamp("phase_rff")

        self.matK = self.gfc.reqMatricePETSc("MatK").reqMat()
        self.matK.assemble()
        self.residu = self.gfc.reqVecteurPETSc("Residu").reqVec()
        self.residu.assemble()

        self.l_matK = []
        
        print("Récupération des pre-post traitements")                        
        self.pp_resolution = self.gfc.reqPP("resolution")
        self.pp_assemblageMatEtRes = self.gfc.reqPP("ppAssMatEtRes")
        self.pp_reinterpole = self.gfc.reqPP("pp_interpoleK")

    def assemblage_premier(self):
        D = self.D_rff 
        l_corr = 0.12
        self.w = np.random.normal(0, 1.0/l_corr, (D, 2))
        
        # Variance de 8% pour éviter les crashs thermiques
        facteur_norm = 0.08 * np.sqrt(1.0 / D)
        
        self.pp_assemblageMatEtRes.execute()
        self.l_matK.append(self.matK.copy())
        
        if self.ordreKL == 0:
            return
            
        for j in range(D):
            self.champ_wx.asgnValeur(float(self.w[j, 0]))
            self.champ_wy.asgnValeur(float(self.w[j, 1]))
            
            # MATRICE COSINUS
            self.champ_phase.asgnValeur(0.0)
            self.pp_reinterpole.execute()
            self.pp_assemblageMatEtRes.execute()
            mat_cos = self.matK.duplicate()
            self.matK.copy(result=mat_cos)
            mat_cos.scale(facteur_norm)
            self.l_matK.append(mat_cos)
            
            # MATRICE SINUS
            self.champ_phase.asgnValeur(-np.pi / 2.0)
            self.pp_reinterpole.execute()
            self.pp_assemblageMatEtRes.execute()
            mat_sin = self.matK.duplicate()
            self.matK.copy(result=mat_sin)
            mat_sin.scale(facteur_norm)
            self.l_matK.append(mat_sin)
              
    def assemblage_second(self):
        dim = self.ordreKL
        # CORRECTION : La matrice bloc doit faire P_total x P_total
        self.K_jk = [[None for _ in range(self.P_total)] for _ in range(self.P_total)]
        
        for j in range(self.P_total):
            for k in range(self.P_total):
                self.K_jk[j][k] = self.matK.duplicate()
                self.K_jk[j][k].zeroEntries()
                
                for i in range(dim+1):
                    coeff = p_chaos.calcul_cijk(i,j,k,dim+1)
                    if abs(coeff) > 1e-12: # Optimisation énorme (on ignore les 0)
                        temp = self.l_matK[i].duplicate()
                        self.l_matK[i].copy(result=temp)
                        temp.scale(coeff)
                        self.K_jk[j][k].axpy(1.0, temp)

                self.K_jk[j][k].assemble()

        self.K_assemble = PETSc.Mat().createNest(self.K_jk)
        self.K_assemble.assemble()

    def assemblage_F(self):
        N = self.residu.getSize()
        liste_construction = [self.residu]

        # On crée P_total - 1 blocs de zéros pour compléter F
        for i in range(self.P_total - 1):
            new_vec = PETSc.Vec().create()
            new_vec.setSizes(N)
            new_vec.setFromOptions()
            new_vec.set(0.0)
            liste_construction.append(new_vec)
            
        self.F_assemble = PETSc.Vec().createNest(liste_construction)
        self.F_assemble.setUp()
        self.F_assemble.assemble()
        
    def construction_T(self):
        N = self.matK.getSize()
        liste_construction = []

        # P_total blocs pour la solution T
        for j in range(self.P_total):
            Tj = PETSc.Vec().createSeq(N)
            Tj.setUp()
            liste_construction.append(Tj)
        
        self.T_assemble = PETSc.Vec().createNest(liste_construction)
        self.T_assemble.setUp()
        self.T_assemble.assemble()

    def resolution_lineaire(self):
        ksp = PETSc.KSP().create()
        ksp.setOperators(self.K_assemble)
        ksp.setTolerances(atol=1e-20, rtol=1e-12, divtol=1e30, max_it=4000)
        ksp.setType('minres')

        def monitor(ksp, its, rnorm):
            # On affiche seulement tous les 10 pour ne pas spammer
            if its % 10 == 0:
                print(f"iteration {its}: residual norm = {rnorm}")
        ksp.setMonitor(monitor)

        try:
            print(f"Taille matrice bloc K : {self.P_total} x {self.P_total} blocs")
            ksp.solve(self.F_assemble, self.T_assemble)

            if ksp.is_converged:
                print(f"convergence en {ksp.getIterationNumber()} iterations")
            else:
                print("La resolution n'a pas converge")
                
            try:
                T0 = self.T_assemble.getNestSubVecs()[0]
                valeurs_T0 = np.array(T0.getArray())
                fichier_ref = "vraie_solution_mc_spatial.npy"
                
                if os.path.exists(fichier_ref):
                    valeurs_exactes = np.load(fichier_ref)
                    if len(valeurs_exactes) == len(valeurs_T0):
                        erreur_max = float(np.max(np.abs(valeurs_T0 - valeurs_exactes)))
                        print(f"\n---> RESULTAT_BENCHMARK_ERREUR={erreur_max:.5f} <---\n")
                else:
                    print(f"RESULTAT_BENCHMARK_ERREUR=FICHIER_REF_INTROUVABLE")
                    
            except Exception as e:
                print(f"Erreur benchmark : {e}")
                
        except PETSc.Error as e:
            print(f"Erreur PETSc : {e}")
            
        return self.T_assemble

    def exporter_statistiques(self):
        """Calcule et exporte la Moyenne (T0) et la Variance totale, et injecte le Vrai MC"""
        print("Calcul et exportation de la Moyenne, de la Variance et du Vrai MC...")
        
        # --- 1. INJECTION DE LA VRAIE SOLUTION SPATIALE (MONTE CARLO) ---
        fichier_ref = "vraie_solution_mc_spatial.npy"
        if os.path.exists(fichier_ref):
            T_mc = np.load(fichier_ref)
            vec_exact = self.gfc.reqVecteurPETSc("T_exacte_vec").reqVec()
            
            # Injection sécurisée et forcée dans PETSc
            indices = np.arange(len(T_mc), dtype=np.int32)
            vec_exact.setValues(indices, T_mc)
            vec_exact.assemble()
            
            # On force MEF++ à mettre à jour le champ géométrique T_exacte_scallin
            self.gfc.reqPP("pp_visu_Texacte").execute()
            print("DEBUG: Transfert vers T_exacte_scallin execute avec succes.")
        else:
            print(f"ATTENTION: Fichier {fichier_ref} introuvable.")

        # --- 2. EXPORT DE L'ESPÉRANCE (MOYENNE SSFEM) ---
        T0 = self.T_assemble.getNestSubVecs()[0]
        self.T_imp = self.gfc.reqVecteurPETSc("T_imp").reqVec()
        T0.copy(result=self.T_imp)
        self.T_imp.assemble()
        
        # Noms de fichiers personnalisés avec D et P
        nom_fichier_moyenne = f"resultats/T_realisation_Moyenne_({self.D_rff},{self.ordrePC})"
        self.gfc.lireLigne(f"""pp_exportation exp_mean [T_ssfem, "{nom_fichier_moyenne}",0,true,false,false,false]""")
        
        # On exécute le transfert PETSc -> Champ (met à jour T_exporte) puis on exporte
        self.gfc.reqPP("pp_visualisation_T").execute()
        self.gfc.reqPP("exp_mean").execute()
        
        # --- 3. EXPORT DE LA VARIANCE SSFEM ---
        T_var = T0.duplicate()
        T_var.set(0.0)
        temp = T0.duplicate()
        
        for j in range(1, self.P_total):
            Tj = self.T_assemble.getNestSubVecs()[j]
            temp.pointwiseMult(Tj, Tj) 
            T_var.axpy(1.0, temp)
            
        T_var.assemble()
        T_var.copy(result=self.T_imp)
        self.T_imp.assemble()
        
        nom_fichier_variance = f"resultats/T_realisation_Variance_({self.D_rff},{self.ordrePC})"
        self.gfc.lireLigne(f"""pp_exportation exp_var [T_ssfem, "{nom_fichier_variance}",0,true,false,false,false]""")
        
        # On met à jour T_exporte avec la variance et on exporte
        self.gfc.reqPP("pp_visualisation_T").execute()
        self.gfc.reqPP("exp_var").execute()
        
        print(f"Fichiers {nom_fichier_moyenne} et {nom_fichier_variance} exportes avec succes ! Ouvre ParaView.")

    def finalise(self):
        mefpp.finalise()

######################  
# Exemple d'utilisation
######################
if len(sys.argv) == 3:
    a = ssfem(int(sys.argv[1]), int(sys.argv[2]))
    a.assemblage_premier()
    a.assemblage_second()
    a.assemblage_F()
    a.construction_T()
    T = a.resolution_lineaire()
    
    # On exporte directement les statistiques (la Moyenne T0 et la Variance)
    a.exporter_statistiques()
    
    a.finalise()