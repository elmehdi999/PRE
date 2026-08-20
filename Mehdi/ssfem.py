########################################################################################################
#Code SSFEM conduction thermique 2D
#Auteur: Augstin PERRIN
########################################################################################################
#Note : Pour récupérer la matrice de Rigidité et du terme source, on utilise un pp codé dans mefpp : ppAssMatEtRes
#Pour récupérer ces matrices, il est nécessaire de ne pas résoudre le problème.
#Pour exporter la solution dans paraview, il est nécessaire de résoudre le problème.
#Donc on met le résidu de côté dès le début du problème.

#Bibliothèques nécessaires pour la simulation des lois aléatoires 
import numpy.random as rd
import numpy as np
#Bibliothèques nécessaires pour la méthode des Éléments finis
import sys
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc
#Bibliothèques développées pour la méthode SSFEM
import kl_expansion 
import p_chaos 
import kl_varphi 
import ks_stat
#Bibliothèque pour la visualisation Monte Carlo
import matplotlib.pyplot as plt

class ssfem:
    #Classe implémentant la méthode SSFEM à partir du fichier conduction.champs
    #Initialisation de mefpp et petsc et des variables de la méthode
    #Implémentation des expansions KL et PC pour la résolution et l'assemblage
    #Assemblage et résolution du système linéaire final (matrices de matrices)
    #Exportation des résultats
    
    def __init__(self,ordreKL,ordrePC):
        #Initialisation de la classe, en fonction de l'ordre de l'expansion KL et PC
        #ordreKL=ordrePC=0 correspond au cas déterministe, FEM sans stochastique
        if ordreKL < ordrePC:                       #En SSFEM, ordreKL<ordrePC impossible
            print("ordreKL < ordrePC impossible")
            # raise ValueError
        elif ordreKL >= 15:                         
            print("OrdreKL trop grand pour l'implémentation actuelle")
            raise ValueError
        
        self.ordreKL = ordreKL
        self.ordrePC = ordrePC
        

        #Initialisation du problème EF et des bibliothèques petsc4py et mefpp4py
        prefixe = "conduction"                  #Mettre nom du fichier .champs
        petsc4py.init(sys.argv)
        mefpp.initialise(prefixe)     #NE PREND PLUS L'ARG pModeTV (Si pModeTV=True , déclenchement de Warning et erreurs qui empêchent l'éxecution)
        mefpp.litEtExecuteActionsDansCollection()
        collection=mefpp.reqCollectionDeCorps()
        collection.lisDonneesDeBase([prefixe])
        corps=collection.reqCorps(prefixe)
        gfc = corps.reqGFC()
        self.gfc = gfc

        #Lecture manuelle des lignes mefpp pour l'expansion KL
        #Pour le moment, expansion jusqu'à ordreKL = 15
        requete_dict = {}
        variable_dict = {}

        for i in range(15):
            requete_dict[i] = "scalaire phi_" + str(i) + " " + kl_varphi.liste_phi_i[i]
            gfc.lireLigne(requete_dict[i])
            variable_dict["K_intermediaire" + str(i)] = "scalaire K_intermediaire2_"+str(i)+" f(K_intermediaire,sqrt_lambda_i, phi_"+str(i)+")=K_intermediaire*2*sqrt_lambda_i*phi_"+ str(i)
            gfc.lireLigne(variable_dict["K_intermediaire" + str(i)])
            variable_dict["pp_reinterpole"+str(i)] = "pp_reinterpole pp_interpoleK_"+str(i)+" [K, K_intermediaire2_"+str(i)+"]"
            gfc.lireLigne(variable_dict["pp_reinterpole" + str(i)])


        #Récupération des champs définis dans le .champs
        self.matK = gfc.reqMatricePETSc("MatK").reqMat()    # Matrice de rigidité "déterministe" à assembler
        self.matK.assemble()
        self.residu = gfc.reqVecteurPETSc("Residu").reqVec()     #Matrice de résidu "déterministe" à assembler
        self.residu.assemble()

        self.lambda_i = gfc.reqChamp("sqrt_lambda_i")       #Coefficients de l'expansion KL

        # Définition des listes de stockage 
        self.l_matK = []  #Liste pour stocker les valeurs de la matrice de rigidité
        
        print("Récupération des pre-post traitements")                        
        self.pp_resolution = gfc.reqPP("resolution")                        #Pour la résolution du problème
        self.pp_assemblageMatEtRes = gfc.reqPP("ppAssMatEtRes")
        self.pp_reinterpole = gfc.reqPP("pp_interpoleK_0")

    def assemblage_premier(self):
        #Méthode de récupération des matrices de rigidité et du résidu
        #En commençant par le cas déterministe, puis en ajoutant les termes de l'expansion

        kl = kl_expansion.kl(np.sqrt(2),0.12,self.ordreKL)
        
        #Apport déterministe
        self.pp_assemblageMatEtRes.execute()
        self.l_matK.append(self.matK.copy())
        self.matK.copy(result=self.l_matK[-1])
        
        if self.ordreKL == 0:
            print("Pas d'expansion KL")
            
        #Apport expansion KL
        for i in range(self.ordreKL):                               #L'expansion commence à lambda_0 et phi_0
            self.lambda_i.asgnValeur(np.sqrt(kl._lambda(i)))        #Mise à la racine ici pour éviter de mettre à la racine dans mefpp
            self.pp_reinterpole = self.gfc.reqPP("pp_interpoleK_"+str(i))
            self.pp_reinterpole.execute()
            self.pp_assemblageMatEtRes.execute()
            self.l_matK.append(self.matK.copy())   
            self.matK.copy(result=self.l_matK[-1])
              
    def assemblage_second(self):
        #Méthode d'assemblage des matrices de rigidité "pseudo-élémentaire" en la matrice de matrices finale

        dim = self.ordreKL                      #Dimension PC = ordre expansion KL
       
        #la matrice sera de taille (ordrePC+1)*(ordrePC+1)
        self.K_jk = [[None for _ in range(self.ordrePC+1)] for _ in range(self.ordrePC+1)]
        
        #Apport expansion PC
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

        #Matrice finale K assemblée
        self.K_assemble = PETSc.Mat().createNest(self.K_jk)
        self.K_assemble.assemble()

    def assemblage_F(self):
        #méthode d'assemblage de la matrice de résidu F finale
        N = self.residu.getSize()

        self.pp_resolution.execute()
        
        liste_construction = [self.residu]  #Liste intermédiaire de construction

        #La matrice est de taille ((ordrePC+1)*degréliberté,degréliberté)
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
        #Méthode de construction du vecteur de solution T finale

        N = self.matK.getSize()    #Degrés de liberté
        liste_construction = []    #Liste intermédiaire de construction

        #Le vecteur est de taille degréliberté*(ordrePC+1)
        for j in range(self.ordrePC+1):
            Tj = PETSc.Vec().createSeq(N)
            Tj.setUp()
            liste_construction.append(Tj)
        
        self.T_assemble = PETSc.Vec().createNest(liste_construction)
        self.T_assemble.setUp()
        self.T_assemble.assemble()

    def resolution_lineaire(self):
        #Méthode de résolution linéaire du problème SSFEM linéaire assemblé final
        #Assemblage_premier(),assemblage_second(),assemblage_F(),construction_T()
        #Doivent avoir été préalablement appelés

        #Déclaration et construction du solveur PETSc
        ksp = PETSc.KSP().create()
        ksp.setOperators(self.K_assemble)
        ksp.setTolerances(atol=1e-20, rtol=1e-12, divtol=1e30, max_it=4000)

        ksp.setType('minres')

        #Pour le moment, pas de préconditionneurs
        #préconditionneur
        # pc = PETSc.PC()
        # pc.create(PETSc.COMM_SELF)
        # pc.setType("cholesky")
        # pc.setOperators(self.K_assemble)
        # ksp.setPC(pc)

        #moniteur de la résolution itérative
        def monitor(ksp, its, rnorm):
            if its <=5 :
                print(f"iteration {its}: residual norm = {rnorm}")
        ksp.setMonitor(monitor)

        #Résolution du problème
        try:
            print(f"Voici taille F : {self.F_assemble.getSize()} et taille T : {self.T_assemble.getSize()}")
            ksp.solve(self.F_assemble, self.T_assemble)

            #Vérification de la convergence
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
        #Méthode de représentation d'une réalisation de T en fonction de T_assemble
        #Et de visualisation de cette réalisation par exportation

        N = self.matK.getSize()
        dim = self.ordreKL 

        self.realisation = PETSc.Vec().createSeq(N)
        self.realisation.setUp()

        #Récupération de la partie déterministe
        T0 = self.T_assemble.getNestSubVecs()[0]  
        T0.copy(result=self.realisation)
        self.realisation.assemble()
        
        liste_tirage = [rd.normal(0,1,dim) for j in range(self.ordrePC)]
        print("liste_tirage :", liste_tirage)
        liste_psi_j = p_chaos.eval_pc_basis(self.ordrePC, dim, liste_tirage)
        for j in range(1,self.ordrePC+1):
            Tj = self.T_assemble.getNestSubVecs()[j]
            temp = Tj.duplicate()
            Tj.copy(result=temp)
            temp.scale(liste_psi_j[j-1])
            self.realisation.axpy(1.0,temp)         #Ajout de la contribution de T_j à la réalisation
            self.realisation.assemble()
            self.realisation.view()
        
        return self.realisation

    def exportation(self):
        """
        Méthode d'exportation de la réalisation courante
        Doit être appelée après resolution_lineaire()
        """

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
        #Méthode de finalisation de la bibliothèque mefpp
        #Doit être executé à la fin du script une seule fois

        print("Finalisation du singleton mefpp")
        mefpp.finalise()

    def mc(self,n_mc,position,affichage=True):
        #Méthode permettant de réaliser des simulations de Monte Carlo
        #Sur la solution T_assemble
        #n_mc = nombre de simulations de Monte Carlo
        #position = numéro du noeud sur lequel on veut récupérer les valeurs de T
        #affichage = booléen pour afficher ou non l'histogramme des résultats de Monte Carlo

        if self.T_assemble is None:
            print("Exécuter la résolution linéaire d'abord")
            return None
        
        liste_realisation = []
        for i in range(n_mc):
            self.realiser()
            T0 = self.realisation
            
            print("On a vu TO")
            liste_realisation.append(float(self.realisation.getValues(position)))  #Récupération de la valeur de T à la position donnée
        print(liste_realisation)
        #Affichaege des résultats de Monte Carlo
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
        #Méthode de test du fit de la solution T_assemble
        #Avec la loi normale par le test de Kolmogorov-Smirnov
        if self.T_assemble is None:
            print("Exécuter la résolution linéaire d'abord")
            return None
        
        #calcul de la moyenne et de la variance à comparer
        liste_echantillon = self.mc(taille_echantillon,position,False)
        moyenne = np.mean(liste_echantillon)
        variance = np.var(liste_echantillon)
        moyenne = self.T_assemble.getNestSubVecs()[0].getValues(position)  #Récupération de la valeur de T à la position donnée
        print(f"Moyenne : {moyenne}, Variance : {variance}")
        #Initialisation de la classe ks_stat
        test_ks = ks_stat.ks(taille_echantillon, moyenne, variance)
        test_ks.setter_echantillon(liste_echantillon)
        calcul = test_ks.kstest()
        test_ks.affichage()


######################  
#Exemple d'utilisation
######################
a = ssfem(int(sys.argv[1]),int(sys.argv[2]))  #sys.argv[1] = ordreKL, sys.argv[2] = ordrePC, mis en place pour faciliter les tests 

#Ordre d'éxecution de la méthode
a.assemblage_premier()
a.assemblage_second()
a.assemblage_F()
a.construction_T()
#Résolution du système et traitement des résultats
T = a.resolution_lineaire()
a.exportation()
a.test_ks(100,64)

