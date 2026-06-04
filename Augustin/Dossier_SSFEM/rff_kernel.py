import numpy as np

class RandomFourierFeatures:
    def __init__(self, dim_spatiale, D, longueur_correlation):
        """
        Initialise l'approximation du noyau gaussien via le théorème de Bochner.
        dim_spatiale : 1, 2 ou 3 (selon le maillage MEF++)
        D : Nombre de fréquences tirées (la précision de l'approximation)
        longueur_correlation : Le paramètre 'l' dans exp(-||x-y||^2 / l^2)
        """
        self.dim = dim_spatiale
        self.D = D
        self.l = longueur_correlation
        
        # La variance des fréquences omega dépend de la longueur de corrélation
        # Pour un noyau gaussien standard, la transformée donne une variance de 2 / l^2
        variance_omega = 2.0 / (self.l ** 2)
        ecart_type_omega = np.sqrt(variance_omega)
        
        # Tirage de la matrice des fréquences aléatoires : taille (D, dim_spatiale)
        # On fixe le seed pour que le champ reste le même à chaque exécution du script
        np.random.seed(42) 
        self.omegas = np.random.normal(0, ecart_type_omega, (self.D, self.dim))

    def eval_Z(self, x):
        """
        Calcule le vecteur trigonométrique Z(x) pour un point spatial donné.
        x : array numpy de coordonnées, ex: [x, y, z]
        Retourne un vecteur 1D de taille 2*D.
        """
        x = np.asarray(x)
        
        # Produit scalaire w_i . x pour toutes les fréquences d'un coup
        # w_x est un vecteur de taille D
        w_x = np.dot(self.omegas, x)
        
        # Calcul du cosinus et sinus pour chaque élément
        cosinus = np.cos(w_x)
        sinus = np.sin(w_x)
        
        # Concaténation et normalisation (comme sur le tableau blanc)
        Z = (1.0 / np.sqrt(self.D)) * np.concatenate((cosinus, sinus))
        
        return Z

    def approx_covariance(self, x, y):
        """
        Approximation de la covariance entre deux points : K(x,y) ≈ Z(x) · Z(y)
        """
        Z_x = self.eval_Z(x)
        Z_y = self.eval_Z(y)
        return np.dot(Z_x, Z_y)

    def vraie_covariance(self, x, y):
        """
        La vraie valeur mathématique du noyau gaussien pour comparer.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        distance_carre = np.sum((x - y)**2)
        return np.exp(-distance_carre / (self.l ** 2))

# ZONE DE TEST (Si on exécute ce fichier directement)

if __name__ == "__main__":
    # Paramètres de test
    dimension = 2       # On teste en 2D !
    precision_D = 500   # On tire 500 fréquences
    longueur_l = 0.5  
    
    rff = RandomFourierFeatures(dimension, precision_D, longueur_l)
    
    # Prenons deux points dans l'espace 2D
    point_A = [0.0, 0.0]
    point_B = [0.5, 0.5]
    
    valeur_approx = rff.approx_covariance(point_A, point_B)
    valeur_exacte = rff.vraie_covariance(point_A, point_B)
    
    print(f"Point A : {point_A} | Point B : {point_B}")
    print(f"Covariance exacte calculée analytiquement : {valeur_exacte:.4f}")
    print(f"Covariance approchée par RFF (D={precision_D})  : {valeur_approx:.4f}")
    print(f"Erreur absolue : {abs(valeur_exacte - valeur_approx):.4e}")