import numpy as np

class RFFField2D:
    """
    Générateur de champs aléatoires Gaussiens 2D basé sur les Random Fourier Features (Théorème de Bochner).
    Remplace l'expansion de Karhunen-Loève.
    """
    def __init__(self, D, longueur_l, sigma=1.0):
        """
        Initialise le champ aléatoire.
        D : dimension de projection (nombre de fréquences). La base finale sera de taille 2*D.
        longueur_l : longueur de corrélation du matériau.
        sigma : écart-type du champ stochastique.
        """
        self.D = D
        self.l = longueur_l
        self.sigma = sigma
        
        # 1. Tirage des fréquences spatiales w
        # Par transformée de Fourier du noyau gaussien, w suit une loi N(0, 1/l^2)
        # On a un problème 2D (x,y), donc w est une matrice de taille (D, 2)
        self.w = np.random.normal(loc=0.0, scale=1.0/self.l, size=(self.D, 2))

    def evaluer_base(self, x, y):
        """
        Évalue les fonctions de base RFF (cos et sin) à un point spatial précis (x, y).
        C'est cette fonction qui va remplacer les phi_i(x) d'Augustin.
        
        Retourne : Un vecteur numpy de taille 2*D.
        """
        pos = np.array([x, y])
        
        # Produit scalaire entre la matrice des fréquences et le vecteur position
        # produit est un vecteur de taille D
        produit = np.dot(self.w, pos)
        
        # Calcul vectoriel ultra-rapide des cosinus et sinus
        base_cos = np.cos(produit)
        base_sin = np.sin(produit)
        
        # Concaténation pour obtenir la base complète [cos(w1.x), sin(w1.x), ..., cos(wD.x), sin(wD.x)]
        # Le facteur de normalisation est issu du théorème de Bochner
        facteur = self.sigma / np.sqrt(self.D)
        base_rff = facteur * np.concatenate((base_cos, base_sin))
        
        return base_rff


# Petit test rapide pour vérifier le module

if __name__ == "__main__":
    # Test avec D=5 et longueur de corrélation l=0.2
    champ = RFFField2D(D=5, longueur_l=0.2)
    
    # On évalue la base spatiale au point géométrique x=0.5, y=1.2
    vecteur_base = champ.evaluer_base(0.5, 1.2)
    
    print(f"Dimension demandée D = {champ.D}")
    print(f"Taille du vecteur de base = {len(vecteur_base)} (car on a cos ET sin)")
    print("Vecteur évalué au point (0.5, 1.2) :")
    print(np.round(vecteur_base, 4))