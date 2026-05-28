#######################################
#Code d'intégration de l'expansion KL
#Dans le cas d'un champ de conductivité
#Aléatoire pour la méthode SSFEM
#Version n°2 dépendant de la seconde
#implémentation KL
#Augustin PERRIN
#GIREF
#######################################


#importation des bibliothèques
import numpy as np
import numpy.random as rd
from scipy.special import factorial
from numpy.polynomial import hermite_e
import matplotlib.pyplot as plt
from scipy import integrate
#définition de la classe


class kl :
    #classe implémentant la troncature de karhunen loeve
    #et différentes méthodes associées

    def __init__(self,w,r,n):

        self.n = n             #indice de troncature : n termes dans l'expansion (ex : ordre 1 = 1 terme constant)

        self.real = []          #realisation courante du champ aléatoire
        self.maillage = [0]

        #paramètres du modèle
        self.w = w              
        self.r = r
        self.alpha = w*np.sqrt(1/(1-r))
        self.beta = w*np.sqrt(r/(1-r**2))
        self.c = ((1+r)/(1-r))**(1/4)
        
        
    def erreur(self,x): 
        #Méthode calculant l'erreur de l'expansion, cf report SUDRET page 109
        err = 1
        for i in range(self.n):
            err -= self._lambda(i)*self._phi(i,x)**2
        return err
    
    def _noyau(self,x,y):
        #Méthode calculant la fonction de covariance 
        return np.exp(-(((x-y)/self.w)**2)/2)
    
    def _lambda(self,n):
        #Méthode calculant \lambda_i de l'expansion KL
        return self.r**n*(1-self.r)
    
    def _hn(self,n,x):
        #Méthode calculant hn(x) le polynôme de Hermite probabiliste d'ordre n
        poly = hermite_e.HermiteE.basis(n)
        return poly(x)/np.sqrt(factorial(n))
    
    def _phi(self,n,x): 
        #Méthode calculant \phi_i de l'expansion KL, non utilisé car kl_varphi.py
        #prend en charge cette fonction pour le .champs
        return self.c * np.exp(-(1/2)*((x/self.alpha)**2))*self._hn(n,x/self.beta)

    def construction_maillage(self,a,b,taille):
        #Méthode de construction du maillage
        self.maillage = np.linspace(a,b,taille)

    def realisation(self,moy=0):
        #Méthode de réalisation d'un champs aléatoire
        #sur un maillage, par l'expansion KL

        H = [moy]*len(self.maillage)

        for n in range(self.n):
            xi = rd.normal(0,1)
            for x in range(len(self.maillage)):
                H[x] += np.sqrt(self._lambda(n))*xi*self._phi(n,self.maillage[x])

        self.real = H.copy()
    
    def affichage(self):
        #Méthode d'affichage de la 
        #réalisation courante
        if self.real == None:
            print("éxecuter : self.realisation() d'abord")
            return None
        print(len(self.maillage), len(self.real))
        plt.plot(self.maillage,self.real,marker='o')
        plt.xlabel("Position des noeuds du maillage")
        plt.ylabel("Valeur en ces noeuds")
        plt.title("Expansion KL champ H")
        plt.show()
        plt.savefig("kl_realisation.png")

    def monte_carlo(self,nmc):
        #méthode de monte carlo sur les réalisations KL
        real_mc = []
        for i in range(nmc):
            self.realisation()
            real_mc.append(self.real.copy())
        return real_mc
    
    def covariance(self,pos1,pos2):
        #méthode de calcul de la covariance 
        #théorique du champ entre pos1 et pos2
        sum = 0
        for i in range(self.n):
            sum+= self._lambda(i)*self._phi(i,pos1)*self._phi(i,pos2)
        return sum
    
    def variance(self):
        #méthode de calcul de la variance 
        #du champ sur le maillage 
        sum = [0] * len(self.maillage)
        for x in range(len(self.maillage)):
            for i in range(self.n):
                sum[x]+= self._lambda(i) * self._phi(i,self.maillage[x])**2
        sum = np.sqrt(sum)
        return sum
    
    def test(self,x,z,n):
        #cette fonction permet de réaliser des tests unitaires
        #sur l'implémentation de KL
        #Ici, test décomposition spectrale
        sum = 0
        for i in range(n):
            sum += self._lambda(i)*self._phi(i,x)*self._phi(i,z)
        
        return sum, self._noyau(x,z)

    def test2(self,n):
        sum = 0
        for i in range(n):
            sum += self._lambda(i)
        return sum
    
##################
#Exemple d'utilisation
##################
# a = kl(1,0.26,10)
# # print(a._lambda(0))
# # print(a._lambda(1))
# a.construction_maillage(-5,5,100)
# # a.realisation()
# # a.realisation()
# print(len(a.variance()))
# print(len(a.maillage))
# plt.plot(a.maillage,a.variance(),label="Variance")
# plt.xlabel("Position des noeuds du maillage")
# plt.ylabel("Variance du champs aléatoire")
# plt.show()
# plt.savefig("kl_expansion_variance_10.png")
