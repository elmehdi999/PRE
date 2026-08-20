import numpy as np

def analyser_support():
    S = np.load("support_sparse.npy")
    print(f"Nombre de polynômes sélectionnés : {len(S)}")
    
    degres = [np.sum(alpha) for alpha in S]
    
    print("\nRépartition par degré (P) :")
    for p in range(6): # De 0 à 5
        compte = degres.count(p)
        print(f" - Degré {p} : {compte} polynômes")
        
    if degres.count(4) == 0 and degres.count(5) == 0:
        print("\n-> CONCLUSION : L'algorithme a totalement ignoré les termes de degré 4 et 5 !")
        print("   Même s'il avait le droit d'en choisir 60, il a préféré prendre des termes de degré 1, 2 et 3.")
        print("   L'ordre P=3 était donc largement suffisant et le plateau de 19% est robuste.")

if __name__ == "__main__":
    analyser_support()