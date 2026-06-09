import subprocess
import time
import csv
import sys
import re
import matplotlib.pyplot as plt

# Configuration des grilles de test
ordres_p = [1, 2, 3]
termes_kl = [1, 2, 5, 10, 14]
frequences_rff = [10, 50, 100, 150, 200, 250, 300, 350]

def verifier_limite_ram(D, p):
    if p == 2 and D > 350: return False
    if p == 3 and D > 250: return False
    return True

def executer_simulation(script_name, param_stochastique, ordre_p):
    cmd = [sys.executable, script_name, str(param_stochastique), str(ordre_p)]
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            erreur_val = "INTROUVABLE"
            # Regex modifiée pour accepter tout ce qui est après le signe "="
            match = re.search(r"RESULTAT_BENCHMARK_ERREUR=(.*)", result.stdout)
            if match:
                erreur_val = match.group(1).strip()
            return elapsed_time, erreur_val, "OK"
        else:
            return None, "", f"Crash (code {result.returncode})"
    except subprocess.TimeoutExpired:
        return None, "", "Timeout"
    except Exception as e:
        return None, "", str(e)

fichier_csv = "resultats_benchmark.csv"

# ==========================================
# 1. PHASE DE CALCUL ET SAUVEGARDE CSV
# ==========================================
with open(fichier_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Methode", "Dimension_Stochastique", "Ordre_PC", "Temps_Secondes", "Erreur_Max", "Statut"])

    print("Début du benchmark KL...")
    for p in ordres_p:
        for m in termes_kl:
            temps, erreur, statut = executer_simulation("ssfem.py", m, p)
            # Affichage console en direct
            if temps:
                print(f"KL  - Modes: {m:2d}, PC: {p} | Temps: {temps:6.2f}s | Erreur_Max: {erreur}")
            else:
                print(f"KL  - Modes: {m:2d}, PC: {p} | Statut: {statut}")
            
            writer.writerow(["KL", m, p, temps if temps else "", erreur, statut])
            f.flush()

    print("\nDébut du benchmark RFF...")
    for p in ordres_p:
        for d in frequences_rff:
            if not verifier_limite_ram(d, p):
                print(f"RFF - D: {d:3d}, PC: {p} | -> Ignoré (Sécurité RAM)")
                writer.writerow(["RFF", d, p, "", "", "Ignore_RAM"])
                continue
                
            temps, erreur, statut = executer_simulation("ssfem_rff.py", d, p)
            # Affichage console en direct
            if temps:
                print(f"RFF - D: {d:3d}, PC: {p} | Temps: {temps:6.2f}s | Erreur_Max: {erreur}")
            else:
                print(f"RFF - D: {d:3d}, PC: {p} | Statut: {statut}")
                
            writer.writerow(["RFF", d, p, temps if temps else "", erreur, statut])
            f.flush()

print("\nCalculs terminés. Génération des graphiques...")

# ==========================================
# 2. PHASE DE POST-TRAITEMENT ET VISUALISATION
# ==========================================
data_kl = {1: {"dim": [], "temps": [], "err": []}, 2: {"dim": [], "temps": [], "err": []}, 3: {"dim": [], "temps": [], "err": []}}
data_rff = {1: {"dim": [], "temps": [], "err": []}, 2: {"dim": [], "temps": [], "err": []}, 3: {"dim": [], "temps": [], "err": []}}

with open(fichier_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Statut"] == "OK" and row["Temps_Secondes"] and row["Erreur_Max"]:
            try:
                methode = row["Methode"]
                dim = int(row["Dimension_Stochastique"])
                p = int(row["Ordre_PC"])
                t = float(row["Temps_Secondes"])
                err = float(row["Erreur_Max"]) # C'est ici que ça plantait avec le texte
                
                if methode == "KL":
                    data_kl[p]["dim"].append(dim)
                    data_kl[p]["temps"].append(t)
                    data_kl[p]["err"].append(err)
                elif methode == "RFF":
                    data_rff[p]["dim"].append(dim)
                    data_rff[p]["temps"].append(t)
                    data_rff[p]["err"].append(err)
            except ValueError:
                print(f"Ligne ignorée pour le tracé ({methode}, param={dim}) : Erreur non lisible '{row['Erreur_Max']}'")
                continue

# --- Figure 1 : Temps d'exécution ---
plt.figure(figsize=(12, 6))
colors = {1: 'blue', 2: 'green', 3: 'red'}
lignes_tracees = 0

for p in ordres_p:
    if data_kl[p]["dim"]:
        plt.plot(data_kl[p]["dim"], data_kl[p]["temps"], marker='o', linestyle='-', color=colors[p], label=f'KL (PC={p})')
        lignes_tracees += 1
    if data_rff[p]["dim"]:
        plt.plot(data_rff[p]["dim"], data_rff[p]["temps"], marker='s', linestyle='--', color=colors[p], label=f'RFF (PC={p})')
        lignes_tracees += 1

if lignes_tracees > 0:
    plt.title("Temps d'exécution en fonction de la dimension stochastique")
    plt.xlabel("Dimension Stochastique (M pour KL, D pour RFF)")
    plt.ylabel("Temps (secondes)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("comparaison_temps.png", dpi=300)
    plt.close()
    print("Graphique 'comparaison_temps.png' généré.")
else:
    print("ATTENTION: Aucune donnée de temps valide pour tracer le graphique.")

# --- Figure 2 : Erreur Maximale ---
plt.figure(figsize=(12, 6))
lignes_tracees_err = 0

for p in ordres_p:
    if data_kl[p]["dim"]:
        plt.plot(data_kl[p]["dim"], data_kl[p]["err"], marker='o', linestyle='-', color=colors[p], label=f'KL (PC={p})')
        lignes_tracees_err += 1
    if data_rff[p]["dim"]:
        plt.plot(data_rff[p]["dim"], data_rff[p]["err"], marker='s', linestyle='--', color=colors[p], label=f'RFF (PC={p})')
        lignes_tracees_err += 1

if lignes_tracees_err > 0:
    plt.title("Erreur maximale (vs T_exact = x) en fonction de la dimension stochastique")
    plt.xlabel("Dimension Stochastique (M pour KL, D pour RFF)")
    plt.ylabel("Erreur Maximale Absolue")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig("comparaison_erreur.png", dpi=300)
    plt.close()
    print("Graphique 'comparaison_erreur.png' généré.")
else:
    print("ATTENTION: Aucune donnée d'erreur valide pour tracer le graphique.")