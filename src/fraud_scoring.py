"""
============================================================================
 PROJET SCORING - DETECTION DE TRANSACTIONS FRAUDULEUSES
============================================================================
Objectif metier : construire un score (probabilite de fraude) permettant de
cibler les transactions frauduleuses parmi les autorisations de paiement par
carte.

Donnees : autorisations.csv (~1,15 M transactions, cible `fraude` ~0,63 %).

Demarche (alignee sur le cours de scoring) :
    1. Chargement et controle qualite des donnees
    2. Analyse exploratoire (EDA) orientee cible
    3. Creation d'indicateurs (feature engineering)
    4. Selection de variables par Weight of Evidence / Information Value
    5. Encodage WoE + separation train / test
    6. Modelisation : regression logistique (scorecard) vs modeles d'arbres
    7. Evaluation (AUC, Gini, KS, PR-AUC, lift) et optimisation du seuil
    8. Modele final, mise a l'echelle du score et interpretation

Le script est concu pour s'executer de bout en bout :
    python scoring_fraude.py
Les figures et tableaux sont sauvegardes dans le dossier `outputs/`.
============================================================================
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend non interactif : on sauvegarde les figures
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATA_PATH = Path("autorisations.csv")
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"

TARGET = "fraude"
RANDOM_STATE = 42
TEST_SIZE = 0.25

# Fraction de l'echantillon a utiliser (1.0 = tout). Permet un test rapide.
SAMPLE_FRAC = float(os.environ.get("SAMPLE_FRAC", "1.0"))

# CodeRep = code reponse de l'autorisation. C'est une information potentiellement
# disponible APRES la decision -> risque de fuite si le score doit etre calcule
# AVANT l'autorisation. On le rend optionnel pour pouvoir comparer les variantes.
USE_CODEREP = True

# Nombre de modalites conservees pour les variables a forte cardinalite
TOP_N_PAYS = 15
TOP_N_MCC = 20

OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# ===========================================================================
# 1. CHARGEMENT ET CONTROLE QUALITE
# ===========================================================================
def charger_donnees() -> pd.DataFrame:
    """Charge le CSV, construit le datetime et controle la qualite de base."""
    print("\n[1] Chargement des donnees ...")
    df = pd.read_csv(DATA_PATH)

    if SAMPLE_FRAC < 1.0:
        df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"    Dimensions          : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
    print(f"    Cartes uniques      : {df['Carte'].nunique():,}")
    print(f"    Taux de fraude      : {100 * df[TARGET].mean():.3f} %  "
          f"({int(df[TARGET].sum()):,} fraudes)")

    # Valeurs manquantes
    na = df.isna().sum()
    na = na[na > 0]
    print(f"    Valeurs manquantes  : "
          f"{'aucune' if na.empty else na.to_dict()}")

    # Construction de l'horodatage (Date au format JJ/MM/AAAA + Heure HH:MM:SS)
    df["horodatage"] = pd.to_datetime(
        df["Date"] + " " + df["Heure"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    n_bad_dt = df["horodatage"].isna().sum()
    if n_bad_dt:
        print(f"    Horodatages invalides : {n_bad_dt}")

    return df


# ===========================================================================
# 2. ANALYSE EXPLORATOIRE
# ===========================================================================
def taux_fraude_par_modalite(df: pd.DataFrame, col: str, top: int = 12) -> pd.DataFrame:
    """Taux de fraude et effectif par modalite d'une variable categorielle."""
    g = (
        df.groupby(col)[TARGET]
        .agg(effectif="count", taux_fraude="mean")
        .sort_values("effectif", ascending=False)
        .head(top)
    )
    g["taux_fraude_%"] = (100 * g["taux_fraude"]).round(3)
    return g.drop(columns="taux_fraude")


def analyse_exploratoire(df: pd.DataFrame) -> None:
    """Produit quelques tableaux et figures descriptives orientees cible."""
    print("\n[2] Analyse exploratoire ...")

    # 2.a Statistiques du montant par classe
    desc = df.groupby(TARGET)["Montant"].describe()[["mean", "50%", "max"]]
    print("    Montant par classe (mean / mediane / max) :")
    print(desc.to_string())

    # 2.b Taux de fraude selon le code reponse
    print("\n    Taux de fraude par CodeRep (top effectifs) :")
    print(taux_fraude_par_modalite(df, "CodeRep").to_string())

    # 2.c Figure : distribution du log-montant par classe
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, sub in df.groupby(TARGET):
        ax.hist(np.log1p(sub["Montant"]), bins=60, alpha=0.5,
                density=True, label=f"fraude={label}")
    ax.set_title("Distribution du log(1+Montant) par classe")
    ax.set_xlabel("log(1 + Montant)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_distribution_montant.png", dpi=120)
    plt.close(fig)

    # 2.d Figure : taux de fraude par heure de la journee
    heure = df["horodatage"].dt.hour
    taux_h = df.groupby(heure)[TARGET].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(taux_h.index, taux_h.values, color="#c0392b")
    ax.set_title("Taux de fraude par heure de la journee")
    ax.set_xlabel("Heure")
    ax.set_ylabel("Taux de fraude (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_fraude_par_heure.png", dpi=120)
    plt.close(fig)

    print(f"    Figures EDA sauvegardees dans {FIG_DIR}/")


# ===========================================================================
# 3. CREATION D'INDICATEURS (FEATURE ENGINEERING)
# ===========================================================================
def regrouper_rares(serie: pd.Series, top_n: int) -> pd.Series:
    """Conserve les `top_n` modalites les plus frequentes, regroupe le reste."""
    tops = serie.value_counts().nlargest(top_n).index
    return serie.where(serie.isin(tops), other="AUTRE").astype(str)


def creer_indicateurs(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les variables explicatives du modele.

    Aucune de ces transformations n'utilise la cible -> pas de fuite, elles
    peuvent etre appliquees avant la separation train/test.
    """
    print("\n[3] Creation d'indicateurs ...")
    d = pd.DataFrame(index=df.index)

    # --- Indicateurs temporels ---
    h = df["horodatage"].dt.hour
    d["heure"] = h
    d["est_nuit"] = ((h >= 0) & (h < 6)).astype(int)          # 0h-6h
    d["jour_semaine"] = df["horodatage"].dt.dayofweek          # 0=lundi
    d["est_weekend"] = (d["jour_semaine"] >= 5).astype(int)

    # --- Montant ---
    d["log_montant"] = np.log1p(df["Montant"].clip(lower=0))
    d["montant_eleve"] = (df["Montant"] > df["Montant"].quantile(0.99)).astype(int)

    # --- Code reponse / refus ---
    d["est_refus"] = (df["CodeRep"] != 0).astype(int)
    if USE_CODEREP:
        # On garde le code regroupe (0 = accepte, sinon top codes de refus)
        d["code_rep"] = regrouper_rares(df["CodeRep"].astype(str), top_n=8)

    # --- Pays et MCC a forte cardinalite : regroupement par frequence ---
    d["pays_grp"] = regrouper_rares(df["Pays"].astype(str), TOP_N_PAYS)
    d["mcc_grp"] = regrouper_rares(df["MCC"].astype(str), TOP_N_MCC)

    # --- Variables metier deja calculees (velocite, cumul, redondance, ecart pays) ---
    fm_cols = [c for c in df.columns if c.startswith("FM_")]
    for c in fm_cols:
        d[c] = df[c].astype(float)
    # Les montants cumules sont tres asymetriques -> passage au log
    for c in [c for c in fm_cols if c.startswith("FM_Sum")]:
        d[c] = np.log1p(d[c].clip(lower=0))

    d[TARGET] = df[TARGET].values
    print(f"    {d.shape[1] - 1} variables explicatives construites.")
    return d


# ===========================================================================
# 4. WEIGHT OF EVIDENCE / INFORMATION VALUE
# ===========================================================================
def woe_iv_from_groups(groups: pd.Series, y: pd.Series, eps: float = 0.5):
    """Calcule la table WoE et l'IV a partir d'une variable deja regroupee.

    Convention : WoE = ln(%non-fraude / %fraude). Un WoE eleve => moins risque.
    Le lissage `eps` evite les log(0) sur les classes sans fraude.
    """
    tab = pd.DataFrame({"g": groups.values, "y": y.values})
    agg = tab.groupby("g")["y"].agg(["sum", "count"])
    agg["bad"] = agg["sum"]                 # fraudes
    agg["good"] = agg["count"] - agg["sum"]  # non-fraudes
    n_bins = len(agg)
    agg["dist_bad"] = (agg["bad"] + eps) / (agg["bad"].sum() + eps * n_bins)
    agg["dist_good"] = (agg["good"] + eps) / (agg["good"].sum() + eps * n_bins)
    agg["woe"] = np.log(agg["dist_good"] / agg["dist_bad"])
    iv = float(((agg["dist_good"] - agg["dist_bad"]) * agg["woe"]).sum())
    return agg["woe"].to_dict(), iv


class WoEEncoder:
    """Encodeur WoE : discretise les variables continues en quantiles, calcule
    le WoE sur le TRAIN uniquement, puis applique le mapping au TRAIN et au TEST.

    Cela garantit l'absence de fuite et fournit en sortie l'IV de chaque
    variable, utilise pour la selection.
    """

    def __init__(self, cont_vars, cat_vars, n_bins=10):
        self.cont_vars = cont_vars
        self.cat_vars = cat_vars
        self.n_bins = n_bins
        self.edges_ = {}     # bornes de discretisation par variable continue
        self.woe_maps_ = {}  # mapping modalite -> WoE
        self.iv_ = {}        # IV par variable

    def _bin_continuous(self, s: pd.Series, var: str, fit: bool) -> pd.Series:
        if fit:
            # Quantiles sur le train ; on borne par -inf / +inf pour le test
            qs = np.unique(np.quantile(s, np.linspace(0, 1, self.n_bins + 1)))
            qs[0], qs[-1] = -np.inf, np.inf
            self.edges_[var] = qs
        return pd.cut(s, bins=self.edges_[var]).astype(str)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for var in self.cont_vars:
            groups = self._bin_continuous(X[var], var, fit=True)
            self.woe_maps_[var], self.iv_[var] = woe_iv_from_groups(groups, y)
        for var in self.cat_vars:
            groups = X[var].astype(str)
            self.woe_maps_[var], self.iv_[var] = woe_iv_from_groups(groups, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for var in self.cont_vars:
            groups = self._bin_continuous(X[var], var, fit=False)
            out[var] = groups.map(self.woe_maps_[var]).fillna(0.0)
        for var in self.cat_vars:
            groups = X[var].astype(str)
            out[var] = groups.map(self.woe_maps_[var]).fillna(0.0)
        return out

    def iv_table(self) -> pd.DataFrame:
        t = pd.Series(self.iv_).sort_values(ascending=False).to_frame("IV")
        # Grille de lecture usuelle de l'Information Value
        t["pouvoir"] = pd.cut(
            t["IV"], [-1, 0.02, 0.1, 0.3, 0.5, np.inf],
            labels=["inutile", "faible", "moyen", "fort", "suspect"],
        )
        return t


# ===========================================================================
# 5. METRIQUES DE SCORING
# ===========================================================================
def ks_statistic(y_true, y_score) -> float:
    """Statistique de Kolmogorov-Smirnov = max separation des distributions."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def best_threshold_f1(y_true, y_score):
    """Seuil maximisant le F1 (utile en contexte tres desequilibre)."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    idx = int(np.argmax(f1[:-1])) if len(thr) else 0
    return float(thr[idx]), float(f1[idx])


def evaluer(nom, y_true, y_score) -> dict:
    """Calcule l'ensemble des metriques de scoring pour un modele."""
    auc = roc_auc_score(y_true, y_score)
    thr, f1 = best_threshold_f1(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "modele": nom,
        "AUC": round(auc, 4),
        "Gini": round(2 * auc - 1, 4),
        "KS": round(ks_statistic(y_true, y_score), 4),
        "PR_AUC": round(average_precision_score(y_true, y_score), 4),
        "seuil*": round(thr, 4),
        "F1*": round(f1, 4),
        "precision*": round(tp / (tp + fp) if (tp + fp) else 0, 4),
        "rappel*": round(tp / (tp + fn) if (tp + fn) else 0, 4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def table_lift(y_true, y_score, n_deciles=10) -> pd.DataFrame:
    """Table de gain/lift : capacite a concentrer la fraude dans les hauts scores."""
    d = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(y_score)})
    d = d.sort_values("s", ascending=False).reset_index(drop=True)
    d["decile"] = pd.qcut(d.index, n_deciles, labels=False) + 1
    taux_global = d["y"].mean()
    g = d.groupby("decile")["y"].agg(effectif="count", fraudes="sum")
    g["taux_fraude_%"] = (100 * g["fraudes"] / g["effectif"]).round(3)
    g["lift"] = (g["fraudes"] / g["effectif"] / taux_global).round(2)
    g["%_fraude_cumule"] = (100 * g["fraudes"].cumsum() / g["fraudes"].sum()).round(1)
    return g


# ===========================================================================
# 6-7. MODELISATION ET EVALUATION
# ===========================================================================
def courbes_roc_pr(resultats_proba: dict, y_true, chemin: Path) -> None:
    """Trace les courbes ROC et Precision-Rappel de tous les modeles."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for nom, score in resultats_proba.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        axes[0].plot(fpr, tpr, label=f"{nom} (AUC={roc_auc_score(y_true, score):.3f})")
        prec, rec, _ = precision_recall_curve(y_true, score)
        axes[1].plot(rec, prec,
                     label=f"{nom} (PR-AUC={average_precision_score(y_true, score):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="Courbe ROC", xlabel="Taux de faux positifs",
                ylabel="Taux de vrais positifs")
    axes[0].legend()
    axes[1].axhline(y_true.mean(), ls="--", c="k", lw=0.8, label="hasard")
    axes[1].set(title="Courbe Precision-Rappel", xlabel="Rappel", ylabel="Precision")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(chemin, dpi=120)
    plt.close(fig)


def modeliser(Xtr, Xte, ytr, yte) -> tuple[pd.DataFrame, dict, dict]:
    """Entraine et evalue plusieurs modeles, retourne les metriques et les scores."""
    print("\n[6] Modelisation ...")

    # Poids pour compenser le desequilibre (~0,63 % de positifs)
    pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    modeles = {
        # (A) Modele de score interpretable : regression logistique sur WoE
        "Regression logistique": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0),
        # (B) Modeles d'arbres pour la performance brute
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=50,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=6,
            class_weight="balanced", random_state=RANDOM_STATE),
    }

    # LightGBM si disponible (present dans l'environnement du cours)
    try:
        from lightgbm import LGBMClassifier
        modeles["LightGBM"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=64,
            scale_pos_weight=pos_weight, n_jobs=-1, random_state=RANDOM_STATE,
            verbose=-1)
    except Exception:
        print("    (LightGBM indisponible : ignore)")

    metriques, proba_test, modeles_fit = [], {}, {}
    for nom, mdl in modeles.items():
        mdl.fit(Xtr, ytr)
        score = mdl.predict_proba(Xte)[:, 1]
        proba_test[nom] = score
        modeles_fit[nom] = mdl
        res = evaluer(nom, yte, score)
        metriques.append(res)
        print(f"    {nom:<22} AUC={res['AUC']:.3f}  Gini={res['Gini']:.3f}  "
              f"KS={res['KS']:.3f}  PR-AUC={res['PR_AUC']:.3f}  F1*={res['F1*']:.3f}")

    tab = pd.DataFrame(metriques).set_index("modele")
    courbes_roc_pr(proba_test, yte, FIG_DIR / "03_courbes_roc_pr.png")
    return tab, proba_test, modeles_fit


# ===========================================================================
# 8. MODELE FINAL : INTERPRETATION ET MISE A L'ECHELLE DU SCORE
# ===========================================================================
def mise_a_echelle_score(proba, pdo=20, score_ref=600, odds_ref=50):
    """Transforme une probabilite de fraude en score points facon scorecard.

    Convention : un score eleve = transaction saine. PDO = points doubling odds.
    """
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    odds_sain = (1 - proba) / proba          # cote en faveur du non-frauduleux
    factor = pdo / np.log(2)
    offset = score_ref - factor * np.log(odds_ref)
    return offset + factor * np.log(odds_sain)


def interpreter_logistique(mdl, feature_names, chemin: Path) -> pd.DataFrame:
    """Coefficients de la regression logistique (sens et force des variables)."""
    coefs = pd.Series(mdl.coef_[0], index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(coefs) + 1))
    coefs.plot(kind="barh", ax=ax, color=np.where(coefs > 0, "#27ae60", "#c0392b"))
    ax.set_title("Coefficients de la regression logistique (sur variables WoE)")
    ax.axvline(0, c="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(chemin, dpi=120)
    plt.close(fig)
    return coefs.to_frame("coefficient")


# ===========================================================================
# PIPELINE PRINCIPAL
# ===========================================================================
def main() -> None:
    df = charger_donnees()
    analyse_exploratoire(df)
    data = creer_indicateurs(df)

    # Definition des types de variables pour l'encodage WoE
    y = data[TARGET]
    cont_vars = (["heure", "log_montant"]
                 + [c for c in data.columns if c.startswith("FM_")])
    cat_vars = ["pays_grp", "mcc_grp", "jour_semaine",
                "est_nuit", "est_weekend", "est_refus", "montant_eleve"]
    if USE_CODEREP:
        cat_vars.append("code_rep")

    # --- Separation train / test stratifiee (preserve le taux de fraude) ---
    X = data.drop(columns=[TARGET])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"\n[5] Train={len(Xtr):,} | Test={len(Xte):,} "
          f"(taux fraude test = {100 * yte.mean():.3f} %)")

    # --- Encodage WoE ajuste sur le train, applique aux deux echantillons ---
    print("\n[4] Encodage WoE et Information Value ...")
    enc = WoEEncoder(cont_vars, cat_vars, n_bins=10).fit(Xtr, ytr)
    iv_tab = enc.iv_table()
    print(iv_tab.round(4).to_string())
    iv_tab.to_csv(OUT_DIR / "information_value.csv")

    # Selection : on retient les variables dont l'IV depasse 0,02 (non inutiles)
    selection = iv_tab[iv_tab["IV"] >= 0.02].index.tolist()
    print(f"\n    Variables retenues (IV >= 0,02) : {len(selection)} / {len(iv_tab)}")

    Xtr_woe = enc.transform(Xtr)[selection]
    Xte_woe = enc.transform(Xte)[selection]

    # --- Modelisation et comparaison ---
    tab, proba_test, modeles_fit = modeliser(Xtr_woe, Xte_woe, ytr, yte)
    tab.to_csv(OUT_DIR / "comparaison_modeles.csv")
    print("\n[7] Comparaison des modeles :")
    print(tab.to_string())

    # --- Selection du meilleur modele (PR-AUC : metrique de reference en fraude) ---
    meilleur = tab["PR_AUC"].idxmax()
    print(f"\n[8] Modele final retenu : {meilleur} (meilleure PR-AUC)")
    score_final = proba_test[meilleur]

    # Table de lift / gain du modele final
    lift = table_lift(yte, score_final)
    lift.to_csv(OUT_DIR / "table_lift.csv")
    print("\n    Table de lift (modele final) :")
    print(lift.to_string())

    # Interpretation de la regression logistique (scorecard interpretable)
    interpreter_logistique(
        modeles_fit["Regression logistique"], selection,
        FIG_DIR / "04_coefficients_logistique.png")

    # Mise a l'echelle du score (0 = tres risque, eleve = sain)
    score_points = mise_a_echelle_score(score_final)
    pd.DataFrame({
        "proba_fraude": np.round(score_final, 5),
        "score_points": np.round(score_points, 1),
        "fraude_reelle": yte.values,
    }).to_csv(OUT_DIR / "scores_test.csv", index=False)

    print(f"\n[OK] Tous les livrables sont dans : {OUT_DIR.resolve()}")
    print("     - information_value.csv   (selection de variables)")
    print("     - comparaison_modeles.csv (metriques par modele)")
    print("     - table_lift.csv          (gain/lift du modele final)")
    print("     - scores_test.csv         (scores individuels)")
    print("     - figures/                (ROC, PR, distributions, coefficients)")


if __name__ == "__main__":
    main()
