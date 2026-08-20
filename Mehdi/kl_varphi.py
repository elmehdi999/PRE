########################################################################################################
#Code d'intégration des fonction \varphi_i de l'expansion KL
#Pour le code SSFEM
#Les fonctions sont implémentées pour pouvoir être lues dans le .champs 
#à l'aide du code python : gfc.lireLigne(...)
#Auteur: Augstin PERRIN
########################################################################################################
#from mefpp4py import mefpp
# import petsc4py
# from petsc4py import PETSc
from scipy.special import factorial
import numpy as np
#modules internes
import kl_expansion
import p_chaos
######################## Définition des phi_i ########################
r = 0.12             #paramètre libre KL
n = 20          #nombre de termes dans kl
r = "0.12"
c = "((1+"+r+")/(1-"+r+"))^(1/4)"
w = "(sqrt(2))"
alpha = w+"*sqrt(1/(1-"+r+"))"
beta =  w+"*sqrt("+r+"/(1-"+r+"^2))"
liste_phi_i = []
hermite = []
# Polynômes de Hermite probabilistes He_n(x) jusqu'à l'ordre 20
hermite.append("1")                                                    # He_0(x)
hermite.append("(x/"+beta+")")                                        # He_1(x)
hermite.append("((x/"+beta+")*(x/"+beta+")-1)")                      # He_2(x)
hermite.append("((x/"+beta+")*(x/"+beta+")*(x/"+beta+")-3*(x/"+beta+"))") # He_3(x)
hermite.append("((x/"+beta+")*(x/"+beta+")*(x/"+beta+")*(x/"+beta+")-6*(x/"+beta+")*(x/"+beta+")+3)") # He_4(x)
hermite.append("((x/"+beta+")^5-10*(x/"+beta+")^3+15*(x/"+beta+"))") # He_5(x)
hermite.append("((x/"+beta+")^6-15*(x/"+beta+")^4+45*(x/"+beta+")^2-15)") # He_6(x)
hermite.append("((x/"+beta+")^7-21*(x/"+beta+")^5+105*(x/"+beta+")^3-105*(x/"+beta+"))") # He_7(x)
hermite.append("((x/"+beta+")^8-28*(x/"+beta+")^6+210*(x/"+beta+")^4-420*(x/"+beta+")^2+105)") # He_8(x)
hermite.append("((x/"+beta+")^9-36*(x/"+beta+")^7+378*(x/"+beta+")^5-1260*(x/"+beta+")^3+945*(x/"+beta+"))") # He_9(x)
hermite.append("((x/"+beta+")^10-45*(x/"+beta+")^8+630*(x/"+beta+")^6-3150*(x/"+beta+")^4+4725*(x/"+beta+")^2-945)") # He_10(x)
hermite.append("((x/"+beta+")^11-55*(x/"+beta+")^9+990*(x/"+beta+")^7-6930*(x/"+beta+")^5+17325*(x/"+beta+")^3-10395*(x/"+beta+"))") # He_11(x)
hermite.append("((x/"+beta+")^12-66*(x/"+beta+")^10+1485*(x/"+beta+")^8-13860*(x/"+beta+")^6+51975*(x/"+beta+")^4-62370*(x/"+beta+")^2+10395)") # He_12(x)
hermite.append("((x/"+beta+")^13-78*(x/"+beta+")^11+2145*(x/"+beta+")^9-25740*(x/"+beta+")^7+135135*(x/"+beta+")^5-270270*(x/"+beta+")^3+135135*(x/"+beta+"))") # He_13(x)
hermite.append("((x/"+beta+")^14-91*(x/"+beta+")^12+3003*(x/"+beta+")^10-45045*(x/"+beta+")^8+315315*(x/"+beta+")^6-945945*(x/"+beta+")^4+945945*(x/"+beta+")^2-135135)") # He_14(x)
# ... continuer le même motif pour les ordres 15-20...

liste_phi_i = [c+"*"+"exp(-(1/2)*(x/"+alpha+")^2)"+"*"+hermite[i] for i in range(len(hermite))]
test_phi_i = ["1","2","3","4","5"]
# print(test_phi_i,"\n")
#Code
l_matK = []


