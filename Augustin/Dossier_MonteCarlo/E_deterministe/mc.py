
#Bibliothèques nécessaires pour la simulation des lois aléatoires
import numpy.random as rd
import numpy as np
#Bibliothèques nécessaires pour la méthode des Éléments finis
import sys
from mefpp4py import mefpp
import petsc4py
from petsc4py import PETSc


class ef_mc:
    def __init__(self):

        print("Initialisation des bibliothèqes")
        petsc4py.init(sys.argv)
        prefixe = "cylindre_cis"                   #A changer en fonction de l'endroit depuis lequel on lance le script
        mefpp.initialise(prefixe,pModeTV=False)  #Si pModeTV=True , déclenchement de Warning et erreurs qui empêchent l'éxecution
        mefpp.litEtExecuteActionsDansCollection()
        collection=mefpp.reqCollectionDeCorps()
        collection.lisDonneesDeBase([prefixe])
        corps=collection.reqCorps(prefixe)
        gfc = corps.reqGFC()

        print("Récupération des champs scalaires définis dans .champs")
        self.theta = gfc.reqChamp("Theta")
        self.phi=gfc.reqChamp("Phi")
        self.rho=gfc.reqChamp("Rho")
        self.vec_u=gfc.reqVecteurPETSc("vec_u").reqVec()
        self.sigma_u=gfc.reqVecteurPETSc("sigma_u").reqVec()
        self.norme_u = gfc.reqVecteurPETSc("norme_u").reqVec()
        self.liste_vec_u = []
        self.liste_sigma_u = []
        self.mefpp_norme_u = []
        #self.liste_u = []
        self.l_theta = []
        self.l_phi = []
        self.l_rho = []
        self.l_cov = []

        print("Récupération des pre-post traitements")
        self.pp_export = gfc.reqPP("exportation_gen2")
        self.pp_export_u = gfc.reqPP("exportation_u")
        self.pp_resolution = gfc.reqPP("resolution")
        self.pp_copie_vec_u = gfc.reqPP("pp_copie_vec_u")
        self.pp_copie_sigma_u = gfc.reqPP("pp_copie_sigma_u")
        self.pp_copie_norme_u = gfc.reqPP("pp_copie_norme_u")
        

    def monte_carlo(self,n_mc):

        for i in range(n_mc):

            self.theta.asgnValeur(0)
            self.phi.asgnValeur(0)
            self.rho.asgnValeur(-5)
            #Résolution de l'equation et exportation des résultats
            self.pp_resolution.execute()
            self.pp_export.execute()
            self.pp_copie_vec_u.execute()
            self.pp_copie_sigma_u.execute()
            self.pp_copie_norme_u.execute()            
            vec_sigma_u = self.sigma_u.getArray()
            self.liste_sigma_u.append(vec_sigma_u.copy())
            #print("Voici la moyenne de stress : ",vec_sigma_u.mean())
            norme_u = self.norme_u.getArray()
            self.mefpp_norme_u.append(norme_u.copy())
            self.pp_export_u.execute()
            vec_copie = self.vec_u.getArray()
            self.liste_vec_u.append(vec_copie.copy())
            #print("Voici la moyenne : ",vec_copie.mean())

    def traitement(self):
        #Traitement des résultats
        #nb_point = 376
        #Les coordonnées des solutions sont stockées dans self.liste_vec_u
        #liste_vec_u[0][1] = u_x au point 1
        #liste_vec_u[0][2] = u_y au point 1
        #liste_vec_u[0][3] = u_z au point 1
        #Il y a 376 points dans le cylindre
        print("Traitement des résultats")
        #self.liste_u = [np.linalg.norm(vec) for vec in self.liste_vec_u]
        u_x,u_y,u_z= [],[],[]
        for i in range(len(self.liste_vec_u)):
            #On extrait les coordonnées u_x, u_y, u_z de chaque solution
            #Chaque solution est un tableau de 1128 valeurs (376 points * 3 coordonnées)
            ux= [self.liste_vec_u[i][3*j] for j in range(len(self.liste_vec_u[0])//3)]
            uy= [self.liste_vec_u[i][3*j+1] for j in range(len(self.liste_vec_u[0])//3)]
            uz= [self.liste_vec_u[i][3*j+2] for j in range(len(self.liste_vec_u[0])//3)]
            u_x.append(ux)
            u_y.append(uy)
            u_z.append(uz)

        #Calcul de la norme de u pour chaque solution
        self.norme_liste_u = [[np.linalg.norm([u_x[i][j], u_y[i][j], u_z[i][j]]) for j in range(len(self.liste_vec_u[0])//3)] for i in range(len(self.liste_vec_u))]
        self.moyenne_norme_u = np.mean(self.norme_liste_u, axis=0)
        self.var_norme_u = np.var(self.norme_liste_u, axis=0)
        #Calcul des moyennes et variance de déformation
        self.moyenne_u_x = np.mean(u_x, axis=0)
        self.moyenne_u_y = np.mean(u_y, axis=0)
        self.moyenne_u_z = np.mean(u_z, axis=0)
   
        self.var_u_x = np.var(u_x, axis=0)  
        self.var_u_y = np.var(u_y, axis=0)
        self.var_u_z = np.var(u_z, axis=0)

        #Calcul du critère COV
        self.cov = [np.sqrt(self.var_norme_u[i])/(np.sqrt(len(self.liste_vec_u)) * self.moyenne_norme_u[i]) for i in range(len(self.liste_vec_u[0])//3) if self.moyenne_norme_u[i] != 0]
        self.l_cov.append(np.max(self.cov))
        #print("Liste des valeurs de norm(u):", self.norme_liste_u)
        # print("Liste des valeurs de theta:", self.l_theta)
        # print("Liste des valeurs de phi:", self.l_phi)
        # print("Liste des valeurs de rho:", self.l_rho)
        # print("Fin du traitement des résultats")


    def affichage_traitement(self):
        print("Affichage des résultats")
        print("Moyenne de u_x:", self.moyenne_u_x)
        print("Moyenne de u_y:", self.moyenne_u_y)
        print("Moyenne de u_z:", self.moyenne_u_z)
        print("Variance de u_x:", self.var_u_x)
        print("Variance de u_y:", self.var_u_y)
        print("Variance de u_z:", self.var_u_z)
        print("Norme des solutions:", self.norme_liste_u)
    
    def critere_cov(self,limite=0.1,niter=200):
        i = 0
        while i<1 or self.l_cov[-1] > limite:
            print("Critère COV non respecté, relance de la simulation")
            print("Critère COV actuel:", self.l_cov[-1])
            self.monte_carlo(1)
            self.traitement()
            i += 1
            if i> niter:
                break
        print("Critère COV respecté, fin de la simulation")
        print("Critère COV final:", min(self.l_cov), "nombre de simulations:", len(self.liste_vec_u))
        
    def finalise(self):
        print("Finalisation du singleton mefpp")
        mefpp.finalise()
    
    # Méthode de comparaison des résultats
    # entre mefpp et les résultats stockés dans liste_norme_u  
    def comparaison(self,e=1e-4):
        a = self.mefpp_norme_u[0]
        b   = self.norme_liste_u[0]
        for i in range(len(a)):
            if abs(a[i] - b[i]) > e:
                print(f"Différence à l'index {i}: {a[i]} != {b[i]}")
                return False
        print("Les listes sont égales dans la tolérance donnée.")
        return True


Bonjour = ef_mc()
Bonjour.monte_carlo(1)
Bonjour.traitement()
Bonjour.affichage_traitement()
