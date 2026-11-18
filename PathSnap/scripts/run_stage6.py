import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.metrics import evaluate_trips_predictions, load_trip_arc_topology
from src.models.model_zoo import PyTorchCandidateSetWrapper

RANDOM_SEED = 42
N_CV_FOLDS = 5
N_OPTUNA_TRIALS = 18


def stack_valid_training_rows(
    trips: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    obs_uid_list: List[np.ndarray] = []
    uid_offset = 0

    for t in trips:
        mask = t["train_row_mask"]
        X_t = t["X_scaled"][mask]
        y_t = t["y"][mask]
        obs_t = t["obs_idx"][mask].astype(np.int64) + uid_offset
        if len(obs_t) > 0:
            uid_offset = int(obs_t.max()) + 1
            X_list.append(X_t)
            y_list.append(y_t)
            obs_uid_list.append(obs_t)

    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(obs_uid_list)


def predict_trip_list(model: PyTorchCandidateSetWrapper, trips: List[Dict]) -> List[np.ndarray]:
    probs_list: List[np.ndarray] = []
    for t in trips:
        p_pos = model.predict_proba(t["X_scaled"], obs_group_ids=t["obs_idx"])[:, 1]
        probs_list.append(p_pos)
    return probs_list


def generate_stage6_plots(
    study: optuna.Study,
    trials_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    vis_dir: Path,
) -> None:
    """Save Optuna optimization history, parameter importance, and Untuned vs. Tuned comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    fig.suptitle(
        "PathSnap Stage 6: Bayesian Hyperparameter Optimisation (CandidateSetResNet)",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1: Optuna Trial Objective History & Best-So-Far Envelope
    t_nums = trials_df["trial_number"].to_numpy()
    objs = trials_df["objective_score"].to_numpy()
    best_so_far = np.maximum.accumulate(objs)
    axes[0].scatter(t_nums, objs, color="#2b5c8f", alpha=0.8, s=45, label="Trial Objective")
    axes[0].plot(t_nums, best_so_far, color="#1b9e3e", linewidth=2.2, label="Best-So-Far Envelope")
    axes[0].set_title("Bayesian TPE Optimization Convergence (5-Fold CV)")
    axes[0].set_xlabel("Trial Number")
    axes[0].set_ylabel("CV Objective (Macro F1 - 0.15 * Obs NLL)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Panel 2: Hyperparameter Importance (fANOVA via Optuna)
    try:
        importances = optuna.importance.get_param_importances(study)
        names = list(importances.keys())[::-1]
        vals = list(importances.values())[::-1]
        axes[1].barh(names, vals, color="#d95f02")
        axes[1].set_title("Hyperparameter Importance (fANOVA)")
        axes[1].set_xlabel("Relative Importance")
        axes[1].grid(axis="x", alpha=0.3)
    except Exception:
        axes[1].text(0.5, 0.5, "Insufficient variance for fANOVA", ha="center", va="center")

    # Panel 3: Untuned vs. Bayesian-Tuned on Student-Test
    metrics_labels = ["Top-1 Acc (%)", "Obs Macro F1 (%)", "Continuity (%)"]
    untuned_vals = [
        comp_df.loc[0, "Student_Test_Top1_Acc"] * 100.0,
        comp_df.loc[0, "Student_Test_Obs_Macro_F1"] * 100.0,
        comp_df.loc[0, "Student_Test_Route_Continuity"] * 100.0,
    ]
    tuned_vals = [
        comp_df.loc[1, "Student_Test_Top1_Acc"] * 100.0,
        comp_df.loc[1, "Student_Test_Obs_Macro_F1"] * 100.0,
        comp_df.loc[1, "Student_Test_Route_Continuity"] * 100.0,
    ]
    x = np.arange(len(metrics_labels))
    w = 0.35
    axes[2].bar(x - w / 2, untuned_vals, w, label="Untuned Baseline", color="#7570b3")
    axes[2].bar(x + w / 2, tuned_vals, w, label="Bayesian Tuned", color="#1b9e3e")
    axes[2].set_title("Untuned vs. Tuned on Student-Test (100%)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(metrics_labels)
    axes[2].set_ylim(75, 100)
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    out_path = vis_dir / "stage6_optuna_optimization.png"
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    processed_dir = ROOT / "data" / "processed"
    artifacts_dir = ROOT / "artifacts"
    logs_dir = ROOT / "logs"
    vis_dir = ROOT / "visualizations"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("PathSnap Stage 6: Bayesian Hyperparameter Optimisation (Optuna TPE)")
    print(f"Selected Model: CandidateSetResNet | Trials: {N_OPTUNA_TRIALS} | CV: 5-Fold GroupKFold")
    print("Strict Rule: Search & CV conducted strictly on 80 Train Trips (eval_test untouched)")
    print("=" * 84)

    train_trips: List[Dict] = joblib.load(processed_dir / "train_features.joblib")
    student_trips: List[Dict] = joblib.load(processed_dir / "student_test_features.joblib")

    train_topo = load_trip_arc_topology(ROOT / "data" / "raw" / "train", [t["trip_id"] for t in train_trips])
    student_topo = load_trip_arc_topology(ROOT / "data" / "raw" / "student_test", [t["trip_id"] for t in student_trips])

    num_features = train_trips[0]["X_scaled"].shape[1]
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    trip_indices = np.arange(len(train_trips))
    trip_groups = np.array([t["trip_id"] for t in train_trips])
    cv_splits = list(gkf.split(trip_indices, groups=trip_groups))

    # Pre-stack fold tensors once so Bayesian trials execute rapidly on GPU
    prepacked_folds = []
    for tr_idx, va_idx in cv_splits:
        tr_trips = [train_trips[i] for i in tr_idx]
        va_trips = [train_trips[i] for i in va_idx]
        X_tr, y_tr, uid_tr = stack_valid_training_rows(tr_trips)
        prepacked_folds.append((X_tr, y_tr, uid_tr, va_trips))

    trial_records: List[Dict] = []

    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        params = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 96, 128, 160]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "lr": trial.suggest_float("lr", 5e-4, 3.5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
            "epochs": trial.suggest_int("epochs", 12, 22),
            "listwise_weight": trial.suggest_float("listwise_weight", 0.50, 0.88),
            "prob_blend_alpha": trial.suggest_float("prob_blend_alpha", 0.30, 0.85),
        }

        fold_metrics: List[Dict] = []
        for fold_idx, (X_tr, y_tr, uid_tr, va_trips) in enumerate(prepacked_folds, start=1):
            est = PyTorchCandidateSetWrapper(
                in_features=num_features,
                hidden_dim=params["hidden_dim"],
                dropout=params["dropout"],
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                listwise_weight=params["listwise_weight"],
                prob_blend_alpha=params["prob_blend_alpha"],
                random_state=RANDOM_SEED + fold_idx,
            )
            est.fit(X_tr, y_tr, obs_group_ids=uid_tr)
            va_probs = predict_trip_list(est, va_trips)
            va_m = evaluate_trips_predictions(va_trips, va_probs, train_topo, use_valid_mask=True)
            fold_metrics.append(va_m)

        cv_top1 = float(np.mean([m["obs_top1_acc"] for m in fold_metrics]))
        cv_f1 = float(np.mean([m["obs_macro_f1"] for m in fold_metrics]))
        cv_obs_ll = float(np.mean([m["obs_logloss"] for m in fold_metrics]))
        cv_pair_ll = float(np.mean([m["pair_logloss"] for m in fold_metrics]))
        cv_cont = float(np.mean([m["route_continuity"] for m in fold_metrics]))

        # Composite objective rewards Macro F1 and calibrated HMM emission log-likelihood
        obj_score = cv_f1 - 0.15 * cv_obs_ll
        elapsed = time.perf_counter() - t0

        rec = {
            "trial_number": trial.number,
            "objective_score": round(obj_score, 6),
            "cv_top1_acc": round(cv_top1, 5),
            "cv_obs_macro_f1": round(cv_f1, 5),
            "cv_obs_logloss": round(cv_obs_ll, 5),
            "cv_pair_logloss": round(cv_pair_ll, 5),
            "cv_route_continuity": round(cv_cont, 5),
            "duration_s": round(elapsed, 2),
            **params,
        }
        trial_records.append(rec)
        print(
            f"  [Trial {trial.number:02d}/{N_OPTUNA_TRIALS-1:02d}] Obj: {obj_score:.5f} | "
            f"CV Top-1: {cv_top1*100:.2f}% | CV F1: {cv_f1:.4f} | Obs NLL: {cv_obs_ll:.4f} | "
            f"dim={params['hidden_dim']}, lr={params['lr']:.4f}, ep={params['epochs']} ({elapsed:.1f}s)"
        )
        return obj_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    trials_df = pd.DataFrame(trial_records)
    trials_csv_path = logs_dir / "stage6_optuna_trials.csv"
    trials_df.to_csv(trials_csv_path, index=False)

    best_params = study.best_params
    best_trial_row = trials_df.loc[trials_df["trial_number"] == study.best_trial.number].iloc[0]

    print("\n" + "-" * 84)
    print(f"[BEST TRIAL #{study.best_trial.number}] Objective: {study.best_value:.5f}")
    print(f"Optimal Hyperparameters: {json.dumps(best_params, indent=2)}")
    print("-" * 84)

    # Refit Tuned CandidateSetResNet on all 80 Training Trips & benchmark on Student-Test
    X_full_tr, y_full_tr, uid_full_tr = stack_valid_training_rows(train_trips)

    fit_t0 = time.perf_counter()
    tuned_model = PyTorchCandidateSetWrapper(
        in_features=num_features,
        hidden_dim=int(best_params["hidden_dim"]),
        dropout=float(best_params["dropout"]),
        lr=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
        batch_size=int(best_params["batch_size"]),
        epochs=int(best_params["epochs"]),
        listwise_weight=float(best_params["listwise_weight"]),
        prob_blend_alpha=float(best_params["prob_blend_alpha"]),
        random_state=RANDOM_SEED,
    )
    tuned_model.fit(X_full_tr, y_full_tr, obs_group_ids=uid_full_tr)
    tuned_fit_s = time.perf_counter() - fit_t0

    inf_t0 = time.perf_counter()
    tuned_stu_probs = predict_trip_list(tuned_model, student_trips)
    tuned_inf_ms = (time.perf_counter() - inf_t0) * 1000.0
    tuned_stu_eval = evaluate_trips_predictions(student_trips, tuned_stu_probs, student_topo, use_valid_mask=False)

    # Load Untuned Stage 4 CandidateSetResNet baseline metrics for side-by-side comparison
    stage4_df = pd.read_csv(logs_dir / "stage4_model_comparison.csv")
    untuned_row = stage4_df.loc[stage4_df["Slug"] == "candidate_set_resnet"].iloc[0]

    comp_df = pd.DataFrame(
        [
            {
                "Configuration": "CandidateSetResNet (Untuned Baseline)",
                "CV_Top1_Acc": untuned_row["CV_Top1_Acc"],
                "CV_Obs_Macro_F1": untuned_row["CV_Obs_Macro_F1"],
                "CV_Obs_LogLoss": untuned_row["CV_Obs_LogLoss"],
                "Student_Test_Top1_Acc": untuned_row["Student_Test_Top1_Acc"],
                "Student_Test_Obs_Macro_F1": untuned_row["Student_Test_Obs_Macro_F1"],
                "Student_Test_Obs_LogLoss": untuned_row["Student_Test_Obs_LogLoss"],
                "Student_Test_Pair_LogLoss": untuned_row["Student_Test_Pair_LogLoss"],
                "Student_Test_Route_Continuity": untuned_row["Student_Test_Route_Continuity"],
                "Student_Test_Inf_Time_ms": untuned_row["Student_Test_Inf_Time_ms"],
            },
            {
                "Configuration": f"CandidateSetResNet (Optuna Trial #{study.best_trial.number})",
                "CV_Top1_Acc": best_trial_row["cv_top1_acc"],
                "CV_Obs_Macro_F1": best_trial_row["cv_obs_macro_f1"],
                "CV_Obs_LogLoss": best_trial_row["cv_obs_logloss"],
                "Student_Test_Top1_Acc": round(tuned_stu_eval["obs_top1_acc"], 5),
                "Student_Test_Obs_Macro_F1": round(tuned_stu_eval["obs_macro_f1"], 5),
                "Student_Test_Obs_LogLoss": round(tuned_stu_eval["obs_logloss"], 5),
                "Student_Test_Pair_LogLoss": round(tuned_stu_eval["pair_logloss"], 5),
                "Student_Test_Route_Continuity": round(tuned_stu_eval["route_continuity"], 5),
                "Student_Test_Inf_Time_ms": round(tuned_inf_ms, 2),
            },
        ]
    )
    comp_csv_path = logs_dir / "stage6_tuned_vs_untuned.csv"
    comp_df.to_csv(comp_csv_path, index=False)

    # Save Tuned Model Checkpoint for Stage 7 (HMM) & Stage 8 (Export)
    ckpt_path = artifacts_dir / "tuned_candidate_set_resnet.pt"
    torch.save(
        {
            "state_dict": tuned_model.model.state_dict(),
            "in_features": num_features,
            "hyperparameters": {
                "hidden_dim": int(best_params["hidden_dim"]),
                "dropout": float(best_params["dropout"]),
                "lr": float(best_params["lr"]),
                "weight_decay": float(best_params["weight_decay"]),
                "batch_size": int(best_params["batch_size"]),
                "epochs": int(best_params["epochs"]),
                "listwise_weight": float(best_params["listwise_weight"]),
                "prob_blend_alpha": float(best_params["prob_blend_alpha"]),
            },
            "best_trial_number": int(study.best_trial.number),
            "best_cv_objective": float(study.best_value),
        },
        ckpt_path,
    )

    generate_stage6_plots(study, trials_df, comp_df, vis_dir)

    print("\n=== STAGE 6: UNTUNED vs. BAYESIAN-TUNED COMPARISON ===")
    print(comp_df.to_string(index=False))
    print(f"\n[SAVED] All Optuna trial logs   -> {trials_csv_path}")
    print(f"[SAVED] Tuned vs Untuned CSV    -> {comp_csv_path}")
    print(f"[SAVED] Tuned PyTorch checkpoint -> {ckpt_path}")
    print(f"[SAVED] Stage 6 Optuna plots    -> {vis_dir / 'stage6_optuna_optimization.png'}")


if __name__ == "__main__":
    main()