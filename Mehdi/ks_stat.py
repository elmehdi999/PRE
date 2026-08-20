#####################
#Importation des bibliothèques
import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.stats as stats
from scipy.stats import norm
##########################"

#Classe implémentant le test de Kolmogorov-Smirnov
class ks:

#Si Dn (valeur table) > K (valeur calculée), 
#l'hypothèse est validée.

    def __init__(self,taille=100, moy=1,variance=0.1):
        #Définition des paramètres
        self.alpha = 0.05
        self.size_sample = taille
        self.echantillon = [0]
        self.mean= moy
        self.var = variance
        #Définition des méthodes 
        self.f_normal =  lambda x: 1/(variance*np.sqrt(2*np.pi))*np.exp(-(x-moy)**2/(2*variance**2))
        #Définition de la table des valeurs
        self.kolmogorov_05 = [None,  # indice 0 inutilisé
    0.975, 0.84189, 0.7076, 0.62394, 0.56328, 0.51926, 0.48342, 0.45427, 0.43001, 0.40925,
    0.39122, 0.37543, 0.36143, 0.34890, 0.33750, 0.32733, 0.31796, 0.30936, 0.30143, 0.29408,
    0.28724, 0.28087, 0.27490, 0.26931, 0.26404, 0.25908, 0.25438, 0.24993, 0.24571, 0.24170,
    0.23788, 0.23424, 0.23076, 0.22743, 0.22425, 0.22119, 0.21826, 0.21544, 0.21273, 0.21012
        ]
        self.f_kolmo = lambda x: 1.358/np.sqrt(x)     # Fonction de la table de Kolmogorov-Smirnov pour alpha = 0.05 et n > 40
    
    def set_param(self,value1,value2):
        #Méthode setter, permettant de modifier alpha et size_sample
        #inutile en pratique
        self.alpha = value1
        self.size_sample = value2

    def setter_echantillon(self, echantillon):
        #Méthode setter, permettant de modifier l'échantillon

        self.echantillon = echantillon
        self.size_sample = len(echantillon)
        self.echantillon.sort()
        
    def tirage_aleatoire(self,mean,var):
        #Méthode de tirage d'un échantillon selon la loi normale théorique, que 
        #la loi expérimentale doit fit
        echantillon_int = []
        for i in range(self.size_sample):
            echantillon_int.append(rd.normal(mean,var))
        echantillon_int.sort()
        self.echantillon = echantillon_int

    def kstest(self):
        #Méthode de test de Kolmogorov-Smirnov
        #On calcule la valeur maximale de |F_n(x) - F(x)| pour x dans l'échantillon
        #et on la compare à la valeur de la table
        #On affiche True si l'hypothèse est validée, False sinon
        calcul_max = 0
        for j in range(self.size_sample-1):
            normal_cdf_j = norm.cdf(self.echantillon[j],loc=self.mean,scale=np.sqrt(self.var))
            normal_cdf_j1 = norm.cdf(self.echantillon[j+1],loc=self.mean,scale=np.sqrt(self.var))
            if np.abs(self.F(self.echantillon[j]) - normal_cdf_j) > calcul_max:
                calcul_max = np.abs(self.F(self.echantillon[j]) - normal_cdf_j)
                # print("Nouveau max : ", calcul_max, " avec indice : ", j, " et [x_j]= ", self.echantillon[j])
            elif np.abs(self.F(self.echantillon[j]) - normal_cdf_j1) > calcul_max:
                calcul_max = np.abs(self.F(self.echantillon[j]) - normal_cdf_j1)
                # print("Nouveau max : ", calcul_max, " avec indice : ", j, " et [x_j]= ", self.echantillon[j])
        if self.size_sample <= 40:
            print("Voici le résultat du test :",calcul_max < self.kolmogorov_05[self.size_sample])
        elif self.size_sample > 40:
            print("Voici le résultat du test :",calcul_max < self.f_kolmo(self.size_sample))
            
        return calcul_max
        
    def F(self,x):
        #Méthode calculant F_n(x) = (nombre d'éléments de l'échantillon <= x)/n
        compteur = 0
        while x > self.echantillon[compteur] and compteur < len(self.echantillon)-1:
            compteur += 1
        return compteur/len(self.echantillon)
        
    def affichage(self):
        #Méthode d'affichage de l'échantillon et de la loi normale théorique
        print(len(self.echantillon))
        #plt.hist(self.echantillon,bins=20,density=True,alpha=0.5)
        plt.plot(self.echantillon,[self.F(self.echantillon[i]) for i in range(self.size_sample)],label="echantillon")
        plt.plot(self.echantillon,[norm.cdf(self.echantillon[i],loc=self.mean,scale=np.sqrt(self.var)) for i in range(self.size_sample)],label="loi normale")
        plt.legend()
        plt.savefig("ks_ssfem_14_10.png")
        plt.show()
    
#####################
#Pour utiliser la classe ks, on peut faire comme suit :
# test = ks(20,0,1)    #pour initialiser avec une taille d'échantillon de 20, une moyenne de 0, une variance de 1 et la loi à comparer
# test.tirage_aleatoire(0,1)      #pour générer un échantillon aléatoire dans
# x = rd.normal(0,1,20)
# test.setter_echantillon(x)  # pour définir l'échantillon à tester
# test.affichage()
# test.kstest()


#####################
#Comparaison avec la 
#bibliothèque scipy
#####################
#x = stats.norm.rvs(size=100, random_state=rng)
#x = rd.uniform(0,1,100)

#stats.kstest(x, stats.norm.cdf)
#g = rd.normal(0,1,20)
#a = [g[i] for i in range(len(g))]
#a.sort()
#print(a)
#Y = [F(-5+0.1*i,a)for i in range(100)]
#plt.plot(np.linspace(-5,5,100),Y,label="echantillon")
#plt.plot(x1,liste_f,label="int f normale")
#plt.legend()
#plt.show()

#def kstest(nb):
#    alpha = 0.05
#    G = tirage_aleatoire(nb)
#    G.sort()
#    calcul_max = 0
#    for j in range(nb):
#        if (j/nb-f_normal(G[j])) > calcul_max :
#            calcul_max = j/nb-f_normal(G[j])
#        elif (f_normal(G[j])-(j-1)/nb) > calcul_max :
#            calcul_max = f_normal(G[j])-(j-1)/nb
#    return calcul_max



