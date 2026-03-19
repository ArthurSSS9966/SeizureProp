import os
import re
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from models import (
    CustomTransformerSeizurePredictor,
    Wavenet,
    ResNet,
    EnhancedResNet,
    load_model_with_config,
)
from utils import map_seizure_channels
from steps import analyze_seizure_propagation
import gc


def discover_pairs(root: str, allowed_patients: List[str] | None = None) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    for pd in sorted(
        d for d in os.listdir(root)
        if d.startswith("P")
        and os.path.isdir(os.path.join(root, d))
        and (allowed_patients is None or d in allowed_patients)
    ):
        pf = os.path.join(root, pd)
        seizes: set[int] = set()
        for n in os.listdir(pf):
            m = re.search(r"seizure_?SZ(\d+)_combined\.pkl$", n, re.IGNORECASE)
            if m:
                seizes.add(int(m.group(1)))
                continue
            m = re.search(r"seizure_?SZ(\d+).*CLEANED\.pkl$", n, re.IGNORECASE)
            if m:
                seizes.add(int(m.group(1)))
        for sz in sorted(seizes):
            pairs.append((pd, sz))
    return pairs


def load_all_models(model_folder: str, device: str) -> Dict[str, torch.nn.Module]:
    names = ["Transformer", "Wavenet", "ResNet", "EnhancedResNet"]
    classes = {
        "Transformer": CustomTransformerSeizurePredictor,
        "Wavenet": Wavenet,
        "ResNet": ResNet,
        "EnhancedResNet": EnhancedResNet,
    }
    loaded: Dict[str, torch.nn.Module] = {}
    for name in names:
        ckpt = os.path.join(model_folder, f"{name}_best.pth")
        if name == "EnhancedResNet" and not os.path.exists(ckpt):
            alt = os.path.join(model_folder, "ResWavenet_best.pth")
            if os.path.exists(alt):
                ckpt = alt
        if not os.path.exists(ckpt):
            print(f"Skip {name}, checkpoint not found at {ckpt}")
            continue
        try:
            model, _ = load_model_with_config(ckpt, classes[name])
            model.to(device).eval()
            print(f"Loaded {name} from {ckpt}")
            loaded[name] = model
        except Exception as e:
            print(f"Error loading {name}: {e}")
    return loaded


def run_eval(
    data_root: str = "data",
    model_folder: str = "checkpoints/BestModels",
    agg_method: str = "mean",
    allowed_patients: List[str] | None = None,
) -> pd.DataFrame:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    models = load_all_models(model_folder, device)
    print("Models:", list(models.keys()))
    pairs = discover_pairs(data_root, allowed_patients)
    print("Total pairs:", len(pairs))

    all_results: Dict[str, List[Dict]] = {}
    per_eval_rows: List[Dict] = []

    for pd_name, sz in pairs:
        pf = os.path.join(data_root, pd_name)
        random_added = False
        for name, model in models.items():
            try:
                results = analyze_seizure_propagation(
                    patient_no=int(pd_name[1:]),
                    seizure_no=sz,
                    model=model,
                    data_folder=data_root,
                    params={"device": device},
                    save_results_ind=False,
                    recalculate_features=False,
                )
            except Exception as e:
                print(f"analyze_seizure_propagation failed for {pd_name} SZ {sz} : {e}")
                continue

            perf = results.get("performance", {})
            with_acc_all = perf.get("accuracy_all_channels", float("nan"))
            with_acc_onset = perf.get("accuracy_onset_channels", float("nan"))

            channel_names = results.get("channel_names", [])
            gt_idx_all = results.get("true_seizure_channels", [])
            gt_idx_onset = results.get("true_onset_channels", [])

            gt_names_all = [channel_names[i] for i in gt_idx_all if i < len(channel_names)]
            gt_names_onset = [channel_names[i] for i in gt_idx_onset if i < len(channel_names)]

            n_all = len(gt_names_all)
            n_onset = len(gt_names_onset)
            top_all = set(channel_names[:n_all]) if n_all > 0 else set()
            top_onset = set(channel_names[:n_onset]) if n_onset > 0 else set()
            no_acc_all = (sum(1 for n in gt_names_all if n in top_all) / n_all) if n_all > 0 else float("nan")
            no_acc_onset = (sum(1 for n in gt_names_onset if n in top_onset) / n_onset) if n_onset > 0 else float("nan")

            try:
                matter_path = os.path.join(pf, "matter.csv")
                matter_df = pd.read_csv(matter_path)
                grey_set = set(matter_df[matter_df["MatterType"].isin(["G", "A"])] ["ElectrodeName"].astype(str))
            except Exception:
                grey_set = set(channel_names)
            pred_masked = [n for n in channel_names if n in grey_set]
            gt_names_all_masked = [n for n in gt_names_all if n in grey_set]

            k = 20
            with_hit20 = (sum(1 for n in gt_names_all_masked if n in set(pred_masked[:k])) / len(gt_names_all_masked)) if len(gt_names_all_masked) > 0 else float("nan")
            no_hit20 = (sum(1 for n in gt_names_all if n in set(channel_names[:k])) / len(gt_names_all)) if len(gt_names_all) > 0 else float("nan")

            all_results.setdefault(name, []).append({
                "patient": pd_name,
                "seizure_no": sz,
                "with_acc_all": with_acc_all,
                "with_acc_onset": with_acc_onset,
                "no_acc_all": no_acc_all,
                "no_acc_onset": no_acc_onset,
                "with_hit20": with_hit20,
                "no_hit20": no_hit20,
            })

            per_eval_rows.append({
                "patient": pd_name,
                "seizure_no": sz,
                "model": name,
                "with_acc_all": with_acc_all,
                "no_acc_all": no_acc_all,
                "with_hit20": with_hit20,
                "no_hit20": no_hit20,
            })

            if not random_added:
                try:
                    rng = np.random.default_rng(hash((pd_name, sz)) & 0xFFFFFFFF)
                    if n_all > 0 and len(channel_names) > 0:
                        top_no = rng.choice(channel_names, size=min(n_all, len(channel_names)), replace=False)
                        rand_no_acc = sum(1 for n in gt_names_all if n in set(top_no)) / n_all
                    else:
                        rand_no_acc = float("nan")
                    if len(gt_names_all_masked) > 0 and len(pred_masked) > 0:
                        top_with = rng.choice(pred_masked, size=min(len(gt_names_all_masked), len(pred_masked)), replace=False)
                        rand_with_acc = sum(1 for n in gt_names_all_masked if n in set(top_with)) / len(gt_names_all_masked)
                    else:
                        rand_with_acc = float("nan")
                    if len(gt_names_all) > 0 and len(channel_names) > 0:
                        top_no20 = rng.choice(channel_names, size=min(k, len(channel_names)), replace=False)
                        rand_no_hit20 = sum(1 for n in gt_names_all if n in set(top_no20)) / len(gt_names_all)
                    else:
                        rand_no_hit20 = float("nan")
                    if len(gt_names_all_masked) > 0 and len(pred_masked) > 0:
                        top_with20 = rng.choice(pred_masked, size=min(k, len(pred_masked)), replace=False)
                        rand_with_hit20 = sum(1 for n in gt_names_all_masked if n in set(top_with20)) / len(gt_names_all_masked)
                    else:
                        rand_with_hit20 = float("nan")

                    all_results.setdefault("Random", []).append({
                        "patient": pd_name,
                        "seizure_no": sz,
                        "with_acc_all": rand_with_acc,
                        "with_acc_onset": float("nan"),
                        "no_acc_all": rand_no_acc,
                        "no_acc_onset": float("nan"),
                        "with_hit20": rand_with_hit20,
                        "no_hit20": rand_no_hit20,
                    })
                    per_eval_rows.append({
                        "patient": pd_name,
                        "seizure_no": sz,
                        "model": "Random",
                        "with_acc_all": rand_with_acc,
                        "no_acc_all": rand_no_acc,
                        "with_hit20": rand_with_hit20,
                        "no_hit20": rand_no_hit20,
                    })
                    random_added = True
                except Exception:
                    pass

            try:
                del results
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    os.makedirs("result", exist_ok=True)
    per_eval_df = pd.DataFrame(per_eval_rows)
    try:
        xlsx_path = os.path.join("result", "gw_prefilter_per_eval.xlsx")
        if os.path.exists(xlsx_path):
            old = pd.read_excel(xlsx_path)
            per_eval_out = pd.concat([old, per_eval_df], ignore_index=True)
        else:
            per_eval_out = per_eval_df
        per_eval_out.to_excel(xlsx_path, index=False)
    except Exception:
        csv_path = os.path.join("result", "gw_prefilter_per_eval.csv")
        if os.path.exists(csv_path):
            old = pd.read_csv(csv_path)
            per_eval_out = pd.concat([old, per_eval_df], ignore_index=True)
        else:
            per_eval_out = per_eval_df
        per_eval_out.to_csv(csv_path, index=False)

    rows = []
    for name, items in all_results.items():
        with_all = [x["with_acc_all"] for x in items if not math.isnan(x["with_acc_all"])]
        no_all = [x["no_acc_all"] for x in items if not math.isnan(x["no_acc_all"])]
        with_h20 = [x["with_hit20"] for x in items if not math.isnan(x["with_hit20"])]
        no_h20 = [x["no_hit20"] for x in items if not math.isnan(x["no_hit20"])]

        def mean_std(a):
            if len(a) == 0:
                return float("nan"), float("nan")
            return float(np.mean(a)), float(np.std(a))

        w_all_m, w_all_s = mean_std(with_all)
        n_all_m, n_all_s = mean_std(no_all)
        w_h_m, w_h_s = mean_std(with_h20)
        n_h_m, n_h_s = mean_std(no_h20)

        rows.append({
            "Model": name,
            "N (acc)": len(with_all),
            "Acc@N With (mean±std)": (f"{w_all_m:.3f}±{w_all_s:.3f}" if not math.isnan(w_all_m) else "NaN"),
            "Acc@N No (mean±std)": (f"{n_all_m:.3f}±{n_all_s:.3f}" if not math.isnan(n_all_m) else "NaN"),
            "ΔAcc@N (With-No)": (w_all_m - n_all_m) if not (math.isnan(w_all_m) or math.isnan(n_all_m)) else float("nan"),
            "N (Hit@20)": len(with_h20),
            "Hit@20 With (mean±std)": (f"{w_h_m:.3f}±{w_h_s:.3f}" if not math.isnan(w_h_m) else "NaN"),
            "Hit@20 No (mean±std)": (f"{n_h_m:.3f}±{n_h_s:.3f}" if not math.isnan(n_h_m) else "NaN"),
            "ΔHit@20 (With-No)": (w_h_m - n_h_m) if not (math.isnan(w_h_m) or math.isnan(n_h_m)) else float("nan"),
        })

    summary_df = pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)
    return summary_df
