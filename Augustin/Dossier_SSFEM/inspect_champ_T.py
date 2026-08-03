import sys
import petsc4py
from mefpp4py import mefpp

def inspecter_champ():
    petsc4py.init(sys.argv)
    mefpp.initialise("conduction_trou")
    mefpp.litEtExecuteActionsDansCollection()
    
    gfc = mefpp.reqCollectionDeCorps().reqCorps("conduction_trou").reqGFC()

    # On récupère l'objet C++
    champ_T = gfc.reqChamp("T")

    print("="*60)
    print(" MÉTHODES DISPONIBLES POUR L'OBJET CHAMP 'T'")
    print("="*60)
    
    # On liste toutes les méthodes publiques de l'objet
    methodes = [m for m in dir(champ_T) if not m.startswith('_')]
    
    for m in methodes:
        print(f" - {m}")
        
    print("="*60)
    
    mefpp.finalise()

if __name__ == "__main__":
    inspecter_champ()