########################################################################################################
# Code SSFEM RFF
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
        self.T_assemble = None 
        
        # calcul de la vraie taille de la base P_total
        self.P_total = math.comb(self.ordreKL + self.ordrePC, self.ordreKL)
        
        print(f"Dimension stochastique (d) : {self.ordreKL}")
        print(f"Ordre du Chaos (P)         : {self.ordrePC}")
        print(f"Taille de la base (P_total): {self.P_total}")
        
        if self.P_total > 500:
            print("ATTENTION : P_total est gigantesque")
            print("Pour tester SSFEM, vous deviez reduire D_rff (ex: 2, 3 ou 4)\n")
        
        prefixe = "conduction_trou"
        petsc4py.init(sys.argv)
        mefpp.initialise(prefixe)
        mefpp.litEtExecuteActionsDansCollection() # resolution du probleme deterministe
        collection=mefpp.reqCollectionDeCorps()
        collection.lisDonneesDeBase([prefixe])
        corps=collection.reqCorps(prefixe)
        self.gfc = corps.reqGFC()

        self.champ_wx = self.gfc.reqChamp("omega_x")
        self.champ_wy = self.gfc.reqChamp("omega_y")
        self.champ_phase = self.gfc.reqChamp("phase_rff")

        self.matK = self.gfc.reqMatricePETSc("MatK").reqMat()
        self.matK.assemble()
        self.residu = self.gfc.reqVecteurPETSc("Residu").reqVec()
        self.residu.assemble()

        self.l_matK = []
        
        print("Recuperation des pre-post traitements")                        
        self.pp_resolution = self.gfc.reqPP("resolution")
        self.pp_assemblageMatEtRes = self.gfc.reqPP("ppAssMatEtRes")
        self.pp_reinterpole = self.gfc.reqPP("pp_interpoleK")

    def assemblage_premier(self):
        D = self.D_rff 
        l_corr = 0.12
        
        # meme variance que monte carlo
        variance_cible = 0.3 
        facteur_norm = variance_cible * np.sqrt(1.0 / D)
        
        rng = np.random.default_rng(42)
        self.w = rng.normal(0, 1.0/l_corr, (D, 2))
        
        # assemblage MEF++ du terme moyen
        self.pp_assemblageMatEtRes.execute()
        self.l_matK.append(self.matK.copy())
        
        if self.ordreKL == 0:
            return
            
        for j in range(D):
            self.champ_wx.asgnValeur(float(self.w[j, 0]))
            self.champ_wy.asgnValeur(float(self.w[j, 1]))
            
            # MATRICE COS
            self.champ_phase.asgnValeur(0.0)
            self.pp_reinterpole.execute()
            self.pp_assemblageMatEtRes.execute()
            mat_cos = self.matK.duplicate()
            self.matK.copy(result=mat_cos)
            mat_cos.axpy(-1.0, self.l_matK[0])  # purification : retire K_mean
            mat_cos.scale(facteur_norm)
            self.l_matK.append(mat_cos)

            # MATRICE SIN
            self.champ_phase.asgnValeur(-np.pi / 2.0)
            self.pp_reinterpole.execute()
            self.pp_assemblageMatEtRes.execute()
            mat_sin = self.matK.duplicate()
            self.matK.copy(result=mat_sin)
            mat_sin.axpy(-1.0, self.l_matK[0])  # purification : retire K_mean
            mat_sin.scale(facteur_norm)
            self.l_matK.append(mat_sin)
              
    def assemblage_second(self):
        dim = self.ordreKL
        self.K_jk = [[None for _ in range(self.P_total)] for _ in range(self.P_total)]
        
        # optimisation par symetrie (k >= j)
        for j in range(self.P_total):
            for k in range(j, self.P_total):
                self.K_jk[j][k] = self.matK.duplicate()
                self.K_jk[j][k].zeroEntries()
                
                for i in range(dim+1):
                    coeff = p_chaos.calcul_cijk(i,j,k,dim+1)
                    if abs(coeff) > 1e-12:
                        temp = self.l_matK[i].duplicate()
                        self.l_matK[i].copy(result=temp)
                        temp.scale(coeff)
                        self.K_jk[j][k].axpy(1.0, temp)

                self.K_jk[j][k].assemble()

                if j != k:
                    self.K_jk[k][j] = self.matK.duplicate()
                    self.K_jk[j][k].copy(result=self.K_jk[k][j])

        self.K_assemble = PETSc.Mat().createNest(self.K_jk)
        self.K_assemble.assemble()

    def assemblage_F(self):
        # on utilise le vecteur residu de MEF++
        N = self.residu.getSize() 
        liste_construction = [self.residu]

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
        N = self.matK.getSize()[0] 
        liste_construction = []

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
            except Exception as e:
                print(f"Erreur benchmark : {e}")
                
        except PETSc.Error as e:
            print(f"Erreur PETSc : {e}")
            
        return self.T_assemble

    def injecter_mc_paraview(self):
        fichier_ref = "vraie_solution_mc_spatial.npy"
        if os.path.exists(fichier_ref):
            T_mc = np.load(fichier_ref)
            vec_exact = self.gfc.reqVecteurPETSc("T_exacte_vec").reqVec()
            
            vec_temp = PETSc.Vec().createSeq(len(T_mc))
            vec_temp.setUp()
            vec_temp.setArray(T_mc)
            vec_temp.assemble()
            vec_temp.copy(result=vec_exact)
            vec_exact.assemble()
            
            self.gfc.reqPP("pp_visu_Texacte").execute()

    def exportation(self):
        self.T_imp = self.gfc.reqVecteurPETSc("T_imp").reqVec() 
        self.realiser().copy(result=self.T_imp)
        self.T_imp.assemble()
        
        self.injecter_mc_paraview()
            
        nom_fichier = f"resultats/T_realisation_({self.D_rff},{self.ordrePC})"
        self.gfc.lireLigne(f"""pp_exportation exportation_T [T_ssfem, "{nom_fichier}",0,true,false,false,false]""")
        self.gfc.reqPP("pp_visualisation_T").execute()
        self.gfc.reqPP("exportation_T").execute()
        print(f"Exportation Realisation : {nom_fichier}.vtu cree.")

    def exporter_statistiques(self):
        print("Calcul et exportation de la Moyenne, de la Variance et de MC...")
        
        self.injecter_mc_paraview()

        T0 = self.T_assemble.getNestSubVecs()[0]
        self.T_imp = self.gfc.reqVecteurPETSc("T_imp").reqVec()
        T0.copy(result=self.T_imp)
        self.T_imp.assemble()
        
        nom_fichier_moyenne = f"resultats/T_Moyenne_({self.D_rff},{self.ordrePC})"
        self.gfc.lireLigne(f"""pp_exportation exp_mean [T_ssfem, "{nom_fichier_moyenne}",0,true,false,false,false]""")
        self.gfc.reqPP("pp_visualisation_T").execute()
        self.gfc.reqPP("exp_mean").execute()
        print(f"Exportation Moyenne : {nom_fichier_moyenne}.vtu cree.")
        
        T_var = T0.duplicate()
        T_var.set(0.0)
        temp = T0.duplicate()
        dim = self.ordreKL
        
        for j in range(1, self.P_total):
            Tj = self.T_assemble.getNestSubVecs()[j]
            temp.pointwiseMult(Tj, Tj) 
            
            norm_j = p_chaos.calcul_cijk(0, j, j, dim + 1)
            T_var.axpy(norm_j, temp)
            
        T_var.assemble()
        T_var.copy(result=self.T_imp)
        self.T_imp.assemble()
        
        nom_fichier_variance = f"resultats/T_Variance_({self.D_rff},{self.ordrePC})"
        self.gfc.lireLigne(f"""pp_exportation exp_var [T_ssfem, "{nom_fichier_variance}",0,true,false,false,false]""")
        self.gfc.reqPP("pp_visualisation_T").execute()
        self.gfc.reqPP("exp_var").execute()
        print(f"Exportation Variance : {nom_fichier_variance}.vtu cree.")

    def realiser(self):
        N = self.matK.getSize()[0]
        dim = self.ordreKL 

        self.realisation = PETSc.Vec().createSeq(N)
        self.realisation.setUp()

        rng_real = np.random.default_rng() 
        xi = rng_real.normal(0, 1, dim)
        
        K_realisation = self.l_matK[0].duplicate()
        self.l_matK[0].copy(result=K_realisation)
        for j in range(dim):
            K_realisation.axpy(xi[j], self.l_matK[j+1])
        K_realisation.assemble()
        
        ksp_mc = PETSc.KSP().create()
        ksp_mc.setOperators(K_realisation)
        ksp_mc.setType('preonly')
        ksp_mc.getPC().setType('lu')
        
        # On utilise le vecteur Residu natif de MEF++
        ksp_mc.solve(self.residu, self.realisation)
        
        return self.realisation

    def mc(self, n_mc, position, affichage=True):
        if self.T_assemble is None:
            print("Veuillez executer la resolution lineaire en premier lieu.")
            return None
        
        liste_realisation = []
        for i in range(n_mc):
            self.realiser()
            liste_realisation.append(float(self.realisation.getValues(position)))
            
        if affichage:
            plt.hist(liste_realisation, bins=30, color='blue')
            plt.xlabel("Valeurs de T")
            plt.ylabel("Densite de probabilite")
            plt.title(f"Histogramme de Monte Carlo pour T a la position {position}")
            plt.grid()
            plt.savefig(f"mc_histo_{position}_kl{self.ordreKL}_pc{self.ordrePC}.png")
            plt.close()
        return liste_realisation

    def test_ks(self, taille_echantillon, position):
        if self.T_assemble is None:
            print("Veuillez executer la resolution lineaire en premier lieu.")
            return None
        
        liste_echantillon = self.mc(taille_echantillon, position, False)
        variance = np.var(liste_echantillon)
        moyenne = self.T_assemble.getNestSubVecs()[0].getValues(position)
        print(f"Moyenne SSFEM : {moyenne:.4f}, Variance Echantillon : {variance:.4f}")
        
        test_ks = ks_stat.ks(taille_echantillon, moyenne, variance)
        test_ks.setter_echantillon(liste_echantillon)
        calcul = test_ks.kstest()
        test_ks.affichage()

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
    
    a.exportation()
    a.exporter_statistiques()
    a.test_ks(100, 500)
    a.finalise()