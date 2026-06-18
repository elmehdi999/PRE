######################
#Code d'implémentation
#du polynomial chaos
######################
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from scipy.special import hermitenorm
from scipy.special import binom
from scipy.special import factorial

#########################
#Fonctions codant l'algorithme récursif
#des balles,
#Nécessaires pour la construction du polynomial chaos
#Et le calcul des coefficients cijk
#Voir SFE SUDRET pour plus de détails
#########################


def algo(M,q):
    """
    Algorithme général de construction des balles
    M : M-1 balles à placer, M est aussi la dimension des variables aléatoires
    q : M+q-1 boîtes, q est le degré du polynôme
    Retourne une liste de listes, chaque liste étant une configuration des balles
    """
    balles=[1 if i<M else 0 for i in range(1,M+q)]
    sequence = [balles.copy()]

    return recursif(balles,sequence)


def find_last_one(balles):
    """
    Trouve la dernière balle (1) dans la liste des balles
    Retourne l'index de la dernière balle (1) ou -1 si aucune balle n'est présente
    """
    try:
        return len(balles) - 1 - balles[::-1].index(1)
    except ValueError:
        return -1


def recursif(balles,sequence):
    """
    Fonction récursive pour générer toutes les configurations possibles des balles
    balles : configuration courante des balles
    sequence : liste pour stocker les configurations générées
    Retourne la liste de toutes les configurations possibles
    """
    M = len(balles)-1
    q = sum(balles)-1
    balle_droite = find_last_one(balles)
    
    if balle_droite!=M and balle_droite >=0: #cas où la derniere balle peut bouger
        # print("Cas 1")
        balles[balle_droite]=0
        balles[balle_droite+1]=1
        sequence.append(balles.copy())
        # print(balles)
        recursif(balles,sequence)

    elif sum([balles[i] for i in range(M,M-(q+1),-1)])!=sum(balles):#cas où une balle autre que la dernière peut bouger
        # print("Cas 2") 
        for j in range(M,-1,-1):

            if balles[j]==0 and balles[j-1]==1:
                balles[j] = 1
                balles[j-1] = 0
                somme = sum([balles[i] for i in range(j,M)])
                balles[j+1::]=[1 if i < somme else 0 for i in range(M-j)]
                sequence.append(balles.copy())
                # print(balles)
                recursif(balles,sequence)
                break

    return sequence


def traduction(balles):
    """
    Fonction de traduction des dispositions des balles
    en la séquence d'entiers correspondante
    balles : liste de 0 et 1 représentant les balles
    Retourne une liste d'entiers représentant les écarts entre les balles
    """
    entiers = []
    curseur = 0
    ecart=0
    while curseur < len(balles):
        
        if balles[curseur] == 1:
            entiers.append(ecart)
            ecart=-1
        curseur+=1
        ecart += 1
    entiers.append(ecart)

    return entiers
 

def alpha_gras(ordreKL,nombre):
    """
    Fonction qui calcule tous les coefficients alpha définissant
    les polynômes de chaos pour une dimension donnée et un ordre donné
    dimension : dimension des variables aléatoires
    nombre : le numéro du polynôme de chaos dont on calcule le coefficiant alpha
    Retourne la liste d'entiers alpha correspondant au polynôme numéro 'nombre'
    """

    p = 0   #ordre à calculer
    somme = 0
    while somme<nombre:
        p+=1 
        somme += binom(ordreKL+p-1,p)
    alpha = algo(ordreKL,p)
    if p > 0 :
        somme_retrait = sum([binom(ordreKL+i-1,ordreKL-1) for i in range(p)])
    else :
        somme_retrait = 0
    return traduction(alpha[int(nombre-somme_retrait)])

def calcul_cijk(i,j,k,ordreKL):
    """
    Fonction calculant le coefficient cijk
    i : indice, i<=ordreKL
    j : numéro du polynôme de chaos j
    k : numéro du polynôme de chaos k
    ordreKL : ordre expansion KL et taille des variables aléatoires
    Retourne le coefficient cijk
    """
    
    alpha_i = alpha_gras(ordreKL,j)
    #print(f"alpha_i : {alpha_i}")
    beta_i = alpha_gras(ordreKL,k)
    
    if i != 0:
        cijk = (alpha_i[i])*(alpha_i[i]-1==beta_i[i]) + (beta_i[i])*(beta_i[i]-1==alpha_i[i])
    
    elif i==0:                            #xi_0 = 1 déterministe
        cijk = factorial(alpha_i[i])*(alpha_i[i]==beta_i[i])


    for l in range(ordreKL):
        if l != i:
            if alpha_i[l] != beta_i[l]:
                cijk = 0
            else:
                cijk *= factorial(alpha_i[l])

    return cijk


def matrice_cijk(i,ordreKL,ordrePC):
    """
    Fonction construisant la matrice cijk
    à des fins de visualisation
    i : indice, i<=ordreKL
    ordreKL : ordre de l'expansion de Karhunen-Loève
    dim : dimension des variables aléatoires
    Retourne la matrice cijk pour l'indice i"""

    if i>ordreKL:
        raise ValueError("i doit être inférieur ou égal à ordreKL")
        
    mat_cijk = np.zeros((ordrePC+1,ordrePC+1))
    for j in range(ordrePC+1):
        for k in range(ordrePC+1):
            
            mat_cijk[j][k] = calcul_cijk(i,j,k,ordreKL)

    return mat_cijk
print(matrice_cijk(0,2,5))
#######################
#Fonctions de calcul des polynômes d'Hermite
#Ces fonctions permettent d'évaluer les polynômes
#PC en des points donnés, pour des alpha donnés.
#Elles ne sont pas utilisées dans le code principal
#mais peuvent être utiles pour des tests ou des visualisations
#######################

def hermite_poly(n,x):
    #Calcule le polynôme d'Hermite probabiliste
    #Utilise scipy.special.hermitenorm
    return hermitenorm(n)(x)

def multi_hermite_poly(alpha,x):
    """
    Calcul du polynôme d'Hermite multivarié, i.e un polynôme de PC
    """
    
    if len(alpha) != len(x):
        raise ValueError("alpha et x doivent avoir la même dimension")
    
    result = 1.0
    for i in range(len(alpha)):
        result *= hermite_poly(alpha[i],x[i])

    return result

def eval_pc_basis(nb_terms, dim, x):
    #Évalue tous les polynômes de chaos jusqu'à un certain ordre

    values = np.zeros(nb_terms)
    for i in range(nb_terms):
        alpha = alpha_gras(dim, i+1)
        values[i] = multi_hermite_poly(alpha,x[i])
    return values

