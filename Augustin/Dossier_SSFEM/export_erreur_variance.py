import numpy as np
import sys
import os
from mefpp4py import mefpp
import petsc4py

def exporter_cartographie_erreur():
    print("="*60)
    print(" EXPORTATION DE L'ERREUR DE VARIANCE SPATIALE")
    print("="*60)

    # 1. Chargement des données existantes
    if not os.path.exists("Y_eval_full_D50_N2048.npy") or not os.path.exists("mc_var_100k.npy"):
        print("Erreur : Fichiers Y_eval_full_D50_N2048.npy ou mc_var_100k.npy introuvables.")
        sys.exit(1)

    data = np.load("Y_eval_full_D50_N2048.npy", allow_pickle=True).item()
    Y_full = data['Y_eval_full']
    var_nisp = np.var(Y_full, axis=0, ddof=1)
    var_mc_100k = np.load("mc_var_100k.npy")

    # Calcul de la différence absolue (pour éviter la division par 0 près de Dirichlet)
    diff_variance = np.abs(var_nisp - var_mc_100k)

    # 2. Initialisation MEF++
    petsc4py.init(sys.argv)
    prefixe = "conduction_trou"
    mefpp.initialise(prefixe)
    mefpp.litEtExecuteActionsDansCollection()
    corps = mefpp.reqCollectionDeCorps().reqCorps(prefixe)
    gfc = corps.reqGFC()

    # 3. Injection dans un vecteur MEF++ pour exportation
    vec_residu = gfc.reqVecteurPETSc("Residu").reqVec()
    indices = np.arange(vec_residu.getSize(), dtype=np.int32)
    
    vec_residu.setValues(indices, diff_variance)
    vec_residu.assemble()

    # Transfert dans le champ T_exporte
    gfc.lireLigne('pp_copie_vecteur_dans_champs pp_push_err [Residu, T_exporte, T]')
    gfc.reqPP("pp_push_err").execute()

    # 4. Exportation VTU
    nom_fichier = "resultats/Cartographie_Erreur_Variance_Absolue"
    gfc.lireLigne(f'pp_exportation exp_err [T_ssfem, "{nom_fichier}",0,true,false,false,false]')
    gfc.reqPP("exp_err").execute()

    print(f"Fichier '{nom_fichier}.vtu' généré.")
    print("Ouvrez ce fichier dans ParaView et affichez 'T_exporte'.")
    print("L'erreur doit être maximale sur le bord extérieur (Dirichlet).")
    
    mefpp.finalise()

if __name__ == "__main__":
    exporter_cartographie_erreur()