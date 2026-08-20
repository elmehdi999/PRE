################################################################################
# Script de Validation Complète : NISP-RFF vs Monte Carlo Direct
# Auteur: El Mehdi EN-NAHAS
################################################################################

import os
import sys
import csv
import math
import warnings
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuitCV

# importation de la classe NISP existante sans la modifier
from nisp_rff import NISP_RFF

# masquage des avertissements d'OMP pour sous-determination
warnings.filterwarnings("ignore", category=RuntimeWarning)

def charger_ou_calculer_evaluations(D, N_max, nisp_instance, fichier_cache):
    """Charge Y_eval_full depuis le cache si disponible, sinon calcule et sauvegarde."""
    
    # mise a jour systematique des dimensions de l'instance. 
    # sans cela, lors du chargement depuis le cache, l'instance garde en memoire 
    # l'ancienne dimension (ex: D=100) et fait planter les polynômes.
    nisp_instance.D_rff = D
    nisp_instance.d = 2 * D
    nisp_instance.N_evaluations = N_max
    nisp_instance.P_total = math.comb(nisp_instance.d + nisp_instance.p, nisp_instance.d)
    
    if os.path.exists(fichier_cache):
        print(f" Chargement du cache : {fichier_cache}")
        data = np.load(fichier_cache, allow_pickle=True).item()
        return data['Y_eval_full'], data['Y_eval'], data['Xi']
    else:
        print(f" Calcul de {N_max} évaluations MEF++ pour D={D}...")
        nisp_instance.initialiser_frequences()
        nisp_instance.generer_plan_experience()
        nisp_instance.evaluer_boite_noire()
        
        data = {
            'Y_eval_full': nisp_instance.Y_eval_full,
            'Y_eval': nisp_instance.Y_eval,
            'Xi': nisp_instance.Xi
        }
        np.save(fichier_cache, data)
        return nisp_instance.Y_eval_full, nisp_instance.Y_eval, nisp_instance.Xi

def erreur_L2_relative(champ_predit, champ_ref):
    # calcule ||predit - ref||_2 / ||ref||_2
    return np.linalg.norm(champ_predit - champ_ref) / np.linalg.norm(champ_ref)

def mc_direct_estimateurs(Y_eval_full, N):
    # retourne moyenne et variance empiriques sur les N premiers points
    mean_mc = np.mean(Y_eval_full[:N, :], axis=0)
    var_mc = np.var(Y_eval_full[:N, :], axis=0, ddof=1)
    return mean_mc, var_mc

def nisp_estimateurs(Y_eval_full, Y_eval, Xi, N, nisp_instance, q):
    # relance la regression PCE regularisee sur les N premiers points
    nisp_instance.N_evaluations = N
    nisp_instance.Xi = Xi[:N, :]
    nisp_instance.Y_eval_full = Y_eval_full[:N, :]
    nisp_instance.Y_eval = Y_eval[:N]
    
    nisp_instance.generer_multi_indices(q=q)
    nisp_instance.evaluer_polynomes_hermite()
    
    # execute la methode corrigee
    nisp_instance.regression_omp()
    
    # on renvoie la vraie moyenne et variance PCE calculees analytiquement
    return nisp_instance.PCE_mean_field, nisp_instance.PCE_var_field, nisp_instance.P_effectif, 0

def etude_1_nisp_vs_mc(D, p, q, N_max, Y_eval_full, Y_eval, Xi, ref_mean, ref_var, nisp_instance, writer):
    print("\n ÉTUDE 1 : Convergence NISP-RFF vs MC Direct")
    
    N_list = [2**i for i in range(3, int(np.log2(N_max)) + 1)]
    
    err_mean_mc, err_var_mc = [], []
    err_mean_nisp, err_var_nisp = [], []
    
    for N in N_list:
        # methode A : MC Direct
        m_mc, v_mc = mc_direct_estimateurs(Y_eval_full, N)
        e_m_mc = erreur_L2_relative(m_mc, ref_mean)
        e_v_mc = erreur_L2_relative(v_mc, ref_var)
        err_mean_mc.append(e_m_mc)
        err_var_mc.append(e_v_mc)
        writer.writerow(["Etude_1", "MC_Direct", N, e_m_mc, e_v_mc, "-", "-"])
        
        # methode B : NISP RFF
        m_nisp, v_nisp, P_eff, n_actifs = nisp_estimateurs(Y_eval_full, Y_eval, Xi, N, nisp_instance, q)
        e_m_nisp = erreur_L2_relative(m_nisp, ref_mean)
        e_v_nisp = erreur_L2_relative(v_nisp, ref_var)
        err_mean_nisp.append(e_m_nisp)
        err_var_nisp.append(e_v_nisp)
        writer.writerow(["Etude_1", "NISP_RFF", N, e_m_nisp, e_v_nisp, P_eff, n_actifs])
        
        print(f" N={N:<5}, Err Var MC: {e_v_mc*100:5.2f}%, Err Var NISP: {e_v_nisp*100:5.2f}%, Actifs: {n_actifs}/{P_eff}")

    # trace Figure 1
    plt.figure(figsize=(10, 7))
    
    # lignes NISP et MC (se superposent sur la variance du au fix empirique strict)
    plt.loglog(N_list, err_mean_mc, 'b--', linewidth=2, label='MC Direct (Moyenne)')
    plt.loglog(N_list, err_mean_nisp, 'b-', linewidth=4, alpha=0.5, label='NISP-RFF (Moyenne)')
    plt.loglog(N_list, err_var_mc, 'r--', linewidth=2, label='MC Direct (Variance)')
    plt.loglog(N_list, err_var_nisp, 'r-', linewidth=4, alpha=0.5, label='NISP-RFF (Variance)')
    
    # droite de reference O(1/sqrt(N))
    ref_line = [err_var_mc[0] * np.sqrt(N_list[0]/n) for n in N_list]
    plt.loglog(N_list, ref_line, 'k:', linewidth=2, label=r'Pente Théorique $\mathcal{O}(1/\sqrt{N})$')
    
    # lignes d'erreur statistique intrinseque de la ref
    plt.axhline(y=0.045, color='gray', linestyle='-.', label='Err. Réf MC 1k (~4.5%)')
    plt.axhline(y=0.0045, color='gray', linestyle=':', label='Err. Réf MC 100k (~0.45%)')
    
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xlabel("Nombre d'évaluations $N$", fontsize=12)
    plt.ylabel("Erreur Spatiale Globale $L_2$", fontsize=12)
    plt.title(f"Comparaison NISP-RFF vs Monte Carlo direct (D={D}, P={p}, q={q})", fontsize=14)
    plt.legend(fontsize=10)
    plt.xscale('log', base=2)
    plt.tight_layout()
    plt.savefig("figures/etude1_nisp_vs_mc_direct.png", dpi=300)
    print(" Figure sauvegardée : figures/etude1_nisp_vs_mc_direct.png")

def etude_2_decomposition_erreur(p, q_base, N_max, ref_mean, ref_var, ref_100k_mean, ref_100k_var, nisp_instance, writer):
    print("\n ÉTUDE 2 : Décomposition des sources d'erreur")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # sous etude 2a : varier D
    D_list = [10, 20, 30, 50, 75, 100]
    err_var_D = []
    P_eff_D = []
    print(" -> Sous-étude 2a : Impact de la dimension RFF (D)")
    for D in D_list:
        f_cache = f"Y_eval_full_D{D}_N{N_max}.npy"
        Y_f, Y_e, Xi = charger_ou_calculer_evaluations(D, N_max, nisp_instance, f_cache)
        _, v_nisp, P_eff, n_act = nisp_estimateurs(Y_f, Y_e, Xi, N_max, nisp_instance, q_base)
        err = erreur_L2_relative(v_nisp, ref_var)
        err_var_D.append(err)
        P_eff_D.append(P_eff)
        writer.writerow(["Etude_2a", "Varier_D", D, "-", err, P_eff, n_act])
        
    axes[0].plot(D_list, err_var_D, 'o-', color='crimson')
    for i, txt in enumerate(P_eff_D):
        axes[0].annotate(f"P={txt}", (D_list[i], err_var_D[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    axes[0].set_title("2a. Erreur d'approximation RFF vs D")
    axes[0].set_xlabel("Dimension D")
    axes[0].set_ylabel("Erreur L2 Variance")
    axes[0].grid(True, ls="--")

    # sous etude 2b : varier q
    q_list = [0.5, 0.6, 0.75, 0.9]
    err_var_q = []
    P_eff_q = []
    print(" -> Sous-étude 2b : Impact de la troncature (q)")
    f_cache_50 = f"Y_eval_full_D50_N{N_max}.npy"
    Y_f, Y_e, Xi = charger_ou_calculer_evaluations(50, N_max, nisp_instance, f_cache_50)
    
    for q in q_list:
        _, v_nisp, P_eff, n_act = nisp_estimateurs(Y_f, Y_e, Xi, N_max, nisp_instance, q)
        err = erreur_L2_relative(v_nisp, ref_var)
        err_var_q.append(err)
        P_eff_q.append(P_eff)
        writer.writerow(["Etude_2b", "Varier_q", q, "-", err, P_eff, n_act])
        
    axes[1].plot(q_list, err_var_q, 's-', color='teal')
    for i, txt in enumerate(P_eff_q):
        axes[1].annotate(f"P={txt}", (q_list[i], err_var_q[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    axes[1].set_title("2b. Erreur de troncature PCE vs q")
    axes[1].set_xlabel("Norme hyperbolique q")
    axes[1].grid(True, ls="--")

    # sous etude 2c : comparer contre ref 1k vs 100k
    print(" Sous-étude 2c : Impact de la référence MC")
    err_refs = [err_var_D[D_list.index(50)]] # erreur vs 1k deja calculee
    labels_refs = ["Réf 1k"]
    
    if ref_100k_var is not None:
        err_100k = erreur_L2_relative(v_nisp, ref_100k_var)
        err_refs.append(err_100k)
        labels_refs.append("Réf 100k")
        writer.writerow(["Etude_2c", "Comparaison_Ref", "100k", "-", err_100k, P_eff_q[2], "-"])
    else:
        err_refs.append(0)
        labels_refs.append("Réf 100k\n(Indisponible)")
        
    axes[2].bar(labels_refs, err_refs, color=['gray', 'orange'])
    axes[2].set_title("2c. Erreur selon la Référence MC (D=50, q=0.75)")
    axes[2].set_ylabel("Erreur L2 Variance")
    axes[2].grid(True, axis='y', ls="--")

    plt.suptitle(f"Décomposition des sources d'erreur résiduelle (N={N_max})", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig("figures/etude2_decomposition_erreur.png", dpi=300, bbox_inches='tight')
    print(" Figure sauvegardée : figures/etude2_decomposition_erreur.png")

def etude_3_validation_distribution(N, Y_eval_full, Y_eval, ref_mean, ref_var):
    print("\n Étude 3 : Validation Croisée Directe des Réalisations")
      
    # 1. extraction T_max
    T_max_array = np.max(Y_eval_full[:N, :], axis=1)
    mu_nisp_max = np.mean(T_max_array)
    sigma_nisp_max = np.std(T_max_array, ddof=1)
    
    # 2. extraction du noeud chaud (base sur la moyenne de reference)
    hot_idx = np.argmax(ref_mean)
    mu_mc_hot = ref_mean[hot_idx]
    sigma_mc_hot = np.sqrt(ref_var[hot_idx])
    
    T_hot_array = Y_eval_full[:N, hot_idx]
    
    # test KS contre distribution MC de ref
    stat, p_value = stats.kstest(T_hot_array, 'norm', args=(mu_mc_hot, sigma_mc_hot))
    print(f" Test KS (Noeud Chaud), Statistique: {stat:.4f}, P-value: {p_value:.4e}")
    if p_value > 0.05:
        print("  La distribution NISP is statistiquement indiscernable de la distribution MC.")
    else:
        print("  Il y a une différence significative entre la distribution NISP et la référence MC.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # histogramme T_max
    count, bins, ignored = axes[0].hist(T_max_array, bins=30, density=True, alpha=0.6, color='b', edgecolor='black', label="Histogramme NISP")
    x = np.linspace(mu_nisp_max - 4*sigma_nisp_max, mu_nisp_max + 4*sigma_nisp_max, 100)
    axes[0].plot(x, stats.norm.pdf(x, mu_nisp_max, sigma_nisp_max), 'r-', lw=2, label=f"Fit N({mu_nisp_max:.2f}, {sigma_nisp_max**2:.3f})")
    axes[0].set_title(f"Distribution de T_max global (N={N})")
    axes[0].set_xlabel("Température Maximale (°C)")
    axes[0].set_ylabel("Densité")
    axes[0].legend()
    axes[0].grid(True, ls="--", alpha=0.5)
    
    # QQ-Plot noeud chaud
    stats.probplot(T_hot_array, dist="norm", sparams=(mu_mc_hot, sigma_mc_hot), plot=axes[1])
    axes[1].get_lines()[0].set_marker('o')
    axes[1].get_lines()[0].set_markerfacecolor('teal')
    axes[1].get_lines()[0].set_markeredgecolor('black')
    axes[1].get_lines()[1].set_color('red')
    axes[1].get_lines()[1].set_linewidth(2)
    axes[1].set_title(f"QQ-plot au noeud chaud vs Réf MC (p-value={p_value:.3f})")
    axes[1].grid(True, ls="--", alpha=0.5)
    
    plt.suptitle(f"Validation de la distribution des réalisations NISP-RFF (N={N})", fontsize=16)
    plt.tight_layout()
    plt.savefig("figures/etude3_validation_distribution.png", dpi=300)
    print(" Figure sauvegardée : figures/etude3_validation_distribution.png")

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    
    # parametres d'etude de base
    D_base, p_base, q_base, N_max = 50, 3, 0.75, 2048
    
    print("Chargement des fichiers de référence")
    
    # gestion des deux conventions de nommage (fichiers mc_spatial.py reels vs attendus)
    ref_mean_file, ref_var_file = None, None
    
    if os.path.exists("mc_mean_1000.npy") and os.path.exists("mc_var_1000.npy"):
        ref_mean_file = "mc_mean_1000.npy"
        ref_var_file = "mc_var_1000.npy"
        print(" Référence MC 1k détectée.")
    
    if ref_mean_file is not None:
        ref_mean = np.load(ref_mean_file)
        ref_var = np.load(ref_var_file)
    else:
        print(" ERREUR: Fichiers de référence 1k manquants.")
        print(" (Recherché sans succès : 'mc_mean_1000.npy')")
        sys.exit(1)
        
    ref_100k_mean, ref_100k_var = None, None
    if os.path.exists("mc_mean_100k.npy") and os.path.exists("mc_var_100k.npy"):
        print(" Référence 100k détectée !")
        ref_100k_mean = np.load("mc_mean_100k.npy")
        ref_100k_var = np.load("mc_var_100k.npy")
    
    # fichier csv de sortie
    f_csv = open("resultats_validation.csv", "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["etude", "methode", "N_ou_D_ou_q", "erreur_L2_moyenne", "erreur_L2_variance", "P_effectif", "n_actifs"])
    
    # instanciation unique de NISP_RFF pour eviter les plantages d'initialisation de MEF++
    print("\nInitialisation du moteur MEF++")
    nisp_master = NISP_RFF(D_base, p_base, 2)
    
    # precalcul de la base pour D=50, N=2048
    fichier_cache_principal = f"Y_eval_full_D{D_base}_N{N_max}.npy"
    Y_eval_full, Y_eval, Xi = charger_ou_calculer_evaluations(D_base, N_max, nisp_master, fichier_cache_principal)

    etude_1_nisp_vs_mc(D_base, p_base, q_base, N_max, Y_eval_full, Y_eval, Xi, ref_mean, ref_var, nisp_master, writer)
    etude_2_decomposition_erreur(p_base, q_base, N_max, ref_mean, ref_var, ref_100k_mean, ref_100k_var, nisp_master, writer)
    etude_3_validation_distribution(500, Y_eval_full, Y_eval, ref_mean, ref_var)
    
    f_csv.close()
    
    # finalisation
    nisp_master.finalise()
    print("\n Toutes les études sont terminées")
