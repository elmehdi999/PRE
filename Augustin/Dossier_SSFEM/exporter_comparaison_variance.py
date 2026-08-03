import numpy as np
import sys
import os
from mefpp4py import mefpp
import petsc4py

def exporter_comparaison_variance():
    print("="*60)
    print(" EXPORTATION 3D DES VARIANCES (MC vs NISP)")
    print("="*60)

    # 1. Chargement des variances
    if not os.path.exists("mc_var_10k.npy") or not os.path.exists("Y_eval_full_D30_N3000.npy"):
        print("Erreur: Les fichiers de variance (mc_var_10k.npy ou le cache NISP) sont introuvables.")
        sys.exit(1)

    var_mc = np.load("mc_var_10k.npy")
    
    # On recalcule la variance NISP à partir du cache
    data_nisp = np.load("Y_eval_full_D30_N3000.npy", allow_pickle=True).item()
    Y_full = data_nisp['Y_eval_full']
    var_nisp = np.var(Y_full, axis=0, ddof=1)

    # On calcule l'écart absolu
    erreur_absolue = np.abs(var_nisp - var_mc)

    # 2. Initialisation MEF++
    petsc4py.init(sys.argv)
    mefpp.initialise("conduction_trou")
    mefpp.litEtExecuteActionsDansCollection()
    gfc = mefpp.reqCollectionDeCorps().reqCorps("conduction_trou").reqGFC()

    # 3. Récupération des vecteurs MEF++
    vec_T = gfc.reqVecteurPETSc("T_imp").reqVec()
    vec_Texacte = gfc.reqVecteurPETSc("T_exacte_vec").reqVec()
    vec_residu = gfc.reqVecteurPETSc("Residu").reqVec()
    indices = np.arange(vec_T.getSize(), dtype=np.int32)

    # 4. Injection des données (en utilisant UNIQUEMENT la numérotation de T)
    
    # Variance MC -> dans le champ T
    vec_T.setValues(indices, var_mc)
    vec_T.assemble()
    gfc.lireLigne('pp_copie_vecteur_dans_champs pp_push_mc [T_imp, T, T]')
    gfc.reqPP("pp_push_mc").execute()

    # Variance NISP -> dans le champ T_exacte_scallin (via pp_visu_Texacte natif)
    vec_Texacte.setValues(indices, var_nisp)
    vec_Texacte.assemble()
    gfc.reqPP("pp_visu_Texacte").execute()

    # Erreur Absolue -> dans le champ T_exporte (via Residu)
    vec_residu.setValues(indices, erreur_absolue)
    vec_residu.assemble()
    gfc.lireLigne('pp_copie_vecteur_dans_champs pp_push_err [Residu, T_exporte, T]')
    gfc.reqPP("pp_push_err").execute()

    # 5. Exportation Personnalisée VTU
    nom_fichier = "resultats/Cartographie_Variances_MC_vs_NISP"
    
    # On crée une configuration d'export sur-mesure pour forcer MEF++
    gfc.lireLigne('exportation exp_var format_exportation VTK')
    gfc.lireLigne('exportation exp_var ajoute_champ T')
    gfc.lireLigne('exportation exp_var ajoute_champ T_exacte_scallin')
    gfc.lireLigne('exportation exp_var ajoute_champ T_exporte')
    
    # Exécution de l'export
    gfc.lireLigne(f'pp_exportation run_exp_var [exp_var, "{nom_fichier}",0,true,false,false,false]')
    gfc.reqPP("run_exp_var").execute()

    print(f"\nFichier '{nom_fichier}.vtu' généré avec succès !")
    print("Ouvrez-le dans ParaView pour comparer :")
    print(" - 'T' : Variance du Monte Carlo (Regardez la chute près du trou !)")
    print(" - 'T_exacte_scallin' : Variance du NISP-RFF (Constante)")
    print(" - 'T_exporte' : L'erreur absolue entre les deux (La fameuse erreur de 20%).")
    
    mefpp.finalise()

if __name__ == "__main__":
    exporter_comparaison_variance()