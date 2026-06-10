########################################################################################################
# Code SSFEM conduction thermique 2D
# Auteur: El Mehdi EN-NAHAS 
########################################################################################################

import numpy.random as rd
import numpy as np
import sys
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc
 
import p_chaos 
import ks_stat
import matplotlib.pyplot as plt

class ssfem:
    def __init__(self, D_rff, ordrePC):
        if D_rff < ordrePC:
            print("Attention : D_rff < ordrePC")
        
        self.D_rff = D_rff  # C'est directement le nombre de vagues (ex: 100)
        self.ordreKL = 2 * D_rff  # La dimension stochastique totale reste le double (cos + sin)
        self.ordrePC = ordrePC
        
        # 1. On pointe vers le nouveau fichier .champs RFF
        prefixe = "conduction_trou"
        petsc4py.init(sys.argv)
        mefpp.initialise(prefixe)
        mefpp.litEtExecuteActionsDansCollection()
        collection=mefpp.reqCollectionDeCorps()
        collection.lisDonneesDeBase([prefixe])
        corps=collection.reqCorps(prefixe)
        gfc = corps.reqGFC()
        self.gfc = gfc

        # 2. PRISE EN CHARGE RFF : On récupère les "télécommandes" pointant vers la mémoire C++
        self.champ_wx = gfc.reqChamp("omega_x")
        self.champ_wy = gfc.reqChamp("omega_y")
        self.champ_phase = gfc.reqChamp("phase_rff") # NOUVEAU

        self.matK = gfc.reqMatricePETSc("MatK").reqMat()
        self.matK.assemble()
        self.residu = gfc.reqVecteurPETSc("Residu").reqVec()
        self.residu.assemble()

        self.l_matK = []
        
        print("Récupération des pre-post traitements")                        
        self.pp_resolution = gfc.reqPP("resolution")
        self.pp_assemblageMatEtRes = gfc.reqPP("ppAssMatEtRes")
        self.pp_reinterpole = gfc.reqPP("pp_interpoleK") # Le "_0" a été supprimé car on a un seul outil dynamique

    def assemblage_premier(self):
        """
        Génération RFF et assemblage des matrices
        """
        D = self.D_rff 
        l_corr = 0.12
        
        # Tirage des fréquences spatiales (D lignes, 2 colonnes)
        self.w = np.random.normal(0, 1.0/l_corr, (D, 2))
        facteur_norm = 1.0 / np.sqrt(D)
        
        # Apport déterministe
        self.pp_assemblageMatEtRes.execute()
        self.l_matK.append(self.matK.copy())
        self.matK.copy(result=self.l_matK[-1])
        
        if self.ordreKL == 0:
            print("Pas d'expansion stochastique")
            return
            
        # Apport stochastique RFF
        for j in range(D):
            # On injecte les nouvelles fréquences
            self.champ_wx.asgnValeur(float(self.w[j, 0]))
            self.champ_wy.asgnValeur(float(self.w[j, 1]))
            
            # --- 1. MATRICE COSINUS ---
            self.champ_phase.asgnValeur(0.0) # phase = 0, c'est un cosinus
            self.pp_reinterpole.execute()
            self.pp_assemblageMatEtRes.execute()
            
            mat_cos = self.matK.duplicate()
            self.matK.copy(result=mat_cos)
            mat_cos.scale(facteur_norm)
            self.l_matK.append(mat_cos)
            
            # --- 2. MATRICE SINUS ---
            self.champ_phase.asgnValeur(-np.pi / 2.0) # phase = -pi/2, cos(x - pi/2) = sin(x)
            self.pp_reinterpole.execute()
            self.pp_assemblageMatEtRes.execute()
            
            mat_sin = self.matK.duplicate()
            self.matK.copy(result=mat_sin)
            mat_sin.scale(facteur_norm)
            self.l_matK.append(mat_sin)
              
    def assemblage_second(self):
        dim = self.ordreKL
        self.K_jk = [[None for _ in range(self.ordrePC+1)] for _ in range(self.ordrePC+1)]
        
        for j in range(self.ordrePC+1):
            for k in range(self.ordrePC+1):
                self.K_jk[j][k] = self.matK.duplicate()
                self.K_jk[j][k].zeroEntries()
                for i in range(dim+1):
                    coeff = p_chaos.calcul_cijk(i,j,k,dim+1)
                    temp = self.l_matK[i].duplicate()
                    self.l_matK[i].copy(result=temp)
                    temp.scale(coeff)
                    self.K_jk[j][k].axpy(1.0,temp)

                self.K_jk[j][k].assemble()

        self.K_assemble = PETSc.Mat().createNest(self.K_jk)
        self.K_assemble.assemble()

    def assemblage_F(self):
        N = self.residu.getSize()
        self.pp_resolution.execute()
        liste_construction = [self.residu]

        for i in range(self.ordrePC):
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

        for j in range(self.ordrePC+1):
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
            if its <=5 :
                print(f"iteration {its}: residual norm = {rnorm}")
        ksp.setMonitor(monitor)

        try:
            print(f"Voici taille F : {self.F_assemble.getSize()} et taille T : {self.T_assemble.getSize()}")
            ksp.solve(self.F_assemble, self.T_assemble)

            if ksp.is_converged:
                print(f"convergence en {ksp.getIterationNumber()} itérations")
                print(f"Résidu final : {ksp.getResidualNorm()}")
            else:
                print("La résolution n'a pas convergé")
                print(f"Résidu final : {ksp.getResidualNorm()}")
            try:
                import numpy as np
                
                T0 = self.T_assemble.getNestSubVecs()[0]
                valeurs_T0 = np.array(T0.getArray())
                
                _gfc = self.gfc if hasattr(self, 'gfc') else gfc
                
                # 1. On lance l'interpolation C++ définie dans le .champs
                _gfc.reqPP("pp_interpoleTexacte").execute()
                
                # 2. On lance la copie C++ vers le vecteur PETSc
                _gfc.reqPP("pp_copie_Texacte_vec").execute()
                
                # 3. Extraction du vecteur exact
                vec_exact = _gfc.reqVecteurPETSc("T_exacte_vec").reqVec()
                vec_exact.assemble()
                valeurs_exactes = np.array(vec_exact.getArray())
                
                if len(valeurs_exactes) == len(valeurs_T0):
                    erreur_max = float(np.max(np.abs(valeurs_T0 - valeurs_exactes)))
                    print(f"RESULTAT_BENCHMARK_ERREUR={erreur_max}")
                else:
                    print(f"RESULTAT_BENCHMARK_ERREUR=ERREUR_TAILLE_PETSC")
                    
            except Exception as e:
                err_name = type(e).__name__
                err_msg = str(e).replace(' ', '_').replace('\n', '')
                print(f"RESULTAT_BENCHMARK_ERREUR=BUG_{err_name}_{err_msg}")
        except PETSc.Error as e:
            print(f"Erreur lors de la résolution : {e}")
            
        return self.T_assemble

    def realiser(self):
        N = self.matK.getSize()
        dim = self.ordreKL 

        self.realisation = PETSc.Vec().createSeq(N)
        self.realisation.setUp()

        T0 = self.T_assemble.getNestSubVecs()[0]  
        T0.copy(result=self.realisation)
        self.realisation.assemble()
        
        liste_tirage = [rd.normal(0,1,dim) for j in range(self.ordrePC)]
        # On peut désactiver le print liste_tirage si c'est trop verbeux
        # print("liste_tirage :", liste_tirage)
        liste_psi_j = p_chaos.eval_pc_basis(self.ordrePC, dim, liste_tirage)
        for j in range(1,self.ordrePC+1):
            Tj = self.T_assemble.getNestSubVecs()[j]
            temp = Tj.duplicate()
            Tj.copy(result=temp)
            temp.scale(liste_psi_j[j-1])
            self.realisation.axpy(1.0,temp)
            self.realisation.assemble()
            self.realisation.view()
        
        return self.realisation

    def exportation(self):
        self.T_imp = self.gfc.reqVecteurPETSc("T_imp").reqVec() 
        self.realiser().copy(result=self.T_imp)
        self.T_imp.assemble()
        self.gfc.lireLigne(f"""pp_exportation exportation_T [T_ssfem, "resultats/T_realisation_({self.ordreKL},{self.ordrePC})",0,true,false,false,false]""")
        pp_copie_Timp = self.gfc.reqPP("pp_visualisation_T")
        pp_exporte = self.gfc.reqPP("exportation_T")
        pp_copie_Timp.execute()
        pp_exporte.execute()
        print("Exportation executée")

    def finalise(self):
        print("Finalisation du singleton mefpp")
        mefpp.finalise()

    def mc(self,n_mc,position,affichage=True):
        if self.T_assemble is None:
            print("Exécuter la résolution linéaire d'abord")
            return None
        
        liste_realisation = []
        for i in range(n_mc):
            self.realiser()
            liste_realisation.append(float(self.realisation.getValues(position)))
            
        if affichage:
            plt.hist(liste_realisation, bins=30, color='blue')
            plt.xlabel("Valeurs de T")
            plt.ylabel("Densité de probabilité")
            plt.title(f"Histogramme de Monte Carlo pour T à la position {position}")
            plt.grid()
            plt.savefig(f"mc_histo_{position}_kl{self.ordreKL}_pc{self.ordrePC}.png")
            plt.close()
        return liste_realisation

    def test_ks(self,taille_echantillon, position):
        if self.T_assemble is None:
            print("Exécuter la résolution linéaire d'abord")
            return None
        
        liste_echantillon = self.mc(taille_echantillon,position,False)
        moyenne = np.mean(liste_echantillon)
        variance = np.var(liste_echantillon)
        moyenne = self.T_assemble.getNestSubVecs()[0].getValues(position)
        print(f"Moyenne : {moyenne}, Variance : {variance}")
        
        test_ks = ks_stat.ks(taille_echantillon, moyenne, variance)
        test_ks.setter_echantillon(liste_echantillon)
        calcul = test_ks.kstest()
        test_ks.affichage()

######################  
# Exemple d'utilisation
######################
if len(sys.argv) == 3:
    # Rappel : sys.argv[1] (ordreKL) doit être un nombre PAIR (ex: 2, 4, 6...)
    a = ssfem(int(sys.argv[1]),int(sys.argv[2]))

    a.assemblage_premier()
    a.assemblage_second()
    a.assemblage_F()
    a.construction_T()
    
    T = a.resolution_lineaire()
    a.exportation()
    a.test_ks(100,64)