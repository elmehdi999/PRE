import subprocess
import os
import numpy as np
import sys
import time

OBJECTIF_FINAL = 10000
TAILLE_BLOC = 100       # MEF++ sera relance toutes les 1000 iterations
FICHIER_ETAT = "mc_state.npy"

while True:
    # lire l'etat actuel pour savoir ou on en est
    iter_faites = 0
    if os.path.exists(FICHIER_ETAT):
        try:
            etat = np.load(FICHIER_ETAT, allow_pickle=True).item()
            iter_faites = etat['iterations_faites']
        except Exception as e:
            print(f"Erreur de lecture de l'état : {e}")
            break
            
    # condition d'arret
    if iter_faites >= OBJECTIF_FINAL:
        print(f"\n Objectif de {OBJECTIF_FINAL} itérations atteint")
        break
        
    # lancer le processus esclave
    iterations_restantes = OBJECTIF_FINAL - iter_faites
    chunk = min(TAILLE_BLOC, iterations_restantes)
    
    print(f"\n Lancement d'un bloc de {chunk} itérations. (Actuel : {iter_faites}/{OBJECTIF_FINAL}) <---")
    
    # execution du script de simulation dans un processus totalement isole
    start_time = time.time()
    result = subprocess.run([sys.executable, "mc_spatial.py", str(chunk)])
    end_time = time.time()
    
    # verification
    if result.returncode != 0:
        print(f"\n Le sous-processus a échoué avec le code {result.returncode}.")
        print("Arrêt du script pour sécurité.")
        sys.exit(1)
        
    print(f" Bloc complété en {end_time - start_time:.2f} s. Purge mémoire RAM effectuée par l'OS.")
    
print("Fin du programme.")