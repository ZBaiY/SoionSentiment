from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .plotting import plot_confusion_matrix, plot_line, plot_pretty_line


_EVAL_SPLITS = {"eval", "val", "valid", "validation", "dev"}


def split_train_events(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_df.empty or "event" not in train_df.columns:
        return train_df, pd.DataFrame()
    event = train_df["event"].astype(str)
    eval_mask = event == "eval"
    train_mask = ~eval_mask
    return train_df[train_mask].copy(), train_df[eval_mask].copy()


def _expand_eval_metrics(eval_events_df: pd.DataFrame) -> pd.DataFrame:
    if eval_events_df.empty or "metrics" not in eval_events_df.columns:
        return eval_events_df
    metrics_flat = pd.json_normalize(eval_events_df["metrics"]).add_prefix("")
    base = eval_events_df.drop(columns=["metrics"])
    return pd.concat([base.reset_index(drop=True), metrics_flat.reset_index(drop=True)], axis=1)


def build_step_axis(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        if cfg.get("strict", {}).get("require_step_axis", True):
            raise RuntimeError("Cannot infer x-axis (empty dataframe)")
        return df

    prefs = cfg.get("analysis", {}).get("x_axis_preference", [])
    for key in prefs:
        if key in df.columns:
            series = pd.to_numeric(df[key], errors="coerce")
            if series.notna().any():
                out = df.copy()
                out["__x_step"] = series
                return out

    try_from = cfg.get("analysis", {}).get("try_construct_step_from", [])
    if len(try_from) >= 2:
        a, b = try_from[0], try_from[1]
        if a in df.columns and b in df.columns:
            epoch = pd.to_numeric(df[a], errors="coerce")
            step_in_epoch = pd.to_numeric(df[b], errors="coerce")
            if epoch.notna().any() and step_in_epoch.notna().any():
                stride = int(step_in_epoch.max()) + 1
                out = df.copy()
                out["__x_step"] = epoch * stride + step_in_epoch
                if not out["__x_step"].is_monotonic_increasing:
                    out = out.sort_values([a, b]).reset_index(drop=True)
                    out["__x_step"] = pd.RangeIndex(start=0, stop=len(out), step=1)
                return out

    if cfg.get("strict", {}).get("require_step_axis", True):
        raise RuntimeError("Cannot infer x-axis (__x_step)")
    return df


def _prefer_eval_split(df: pd.DataFrame) -> pd.DataFrame:
    if "split" not in df.columns:
        return df
    split = df["split"].astype(str).str.lower()
    cand = df[split.isin(_EVAL_SPLITS)]
    if not cand.empty:
        return cand
    return df


def _select_eval_split(df: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    if "split" not in df.columns:
        return df
    split = df["split"].astype(str).str.lower()
    for name in preferred:
        cand = df[split == name]
        if not cand.empty:
            return cand
    return df


def _select_metric_name(df: pd.DataFrame, cfg: dict[str, Any]) -> str | None:
    prefs = cfg.get("analysis", {}).get("best_metric_preference", [])
    for m in prefs:
        if m in df.columns and df[m].notna().any():
            return m
    return None


def compute_best(metrics_df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    if metrics_df.empty:
        return {}

    metric_name = _select_metric_name(metrics_df, cfg)
    if metric_name is None:
        return {}

    df = _prefer_eval_split(metrics_df)
    series = pd.to_numeric(df[metric_name], errors="coerce")
    df = df.copy()
    df[metric_name] = series
    df = df.dropna(subset=[metric_name])
    if df.empty:
        return {}

    if metric_name == "loss":
        row = df.loc[df[metric_name].idxmin()]
    else:
        row = df.loc[df[metric_name].idxmax()]

    out: dict[str, Any] = {
        "metric": metric_name,
        "value": float(row[metric_name]),
    }
    if "__x_step" in row.index:
        out["x_step"] = float(row["__x_step"])
    for key in ["step", "epoch", "split"]:
        if key in row.index and pd.notna(row[key]):
            value = row[key]
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            out[key] = value
    return out


def compute_best_from_eval_events(eval_df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    if eval_df.empty:
        return {}
    metric_name = _select_metric_name(eval_df, cfg)
    if metric_name is None and "eval_loss" in eval_df.columns:
        prefs = cfg.get("analysis", {}).get("best_metric_preference", [])
        if "loss" in prefs:
            metric_name = "eval_loss"
    if metric_name is None:
        return {}
    df = eval_df.copy()
    df[metric_name] = pd.to_numeric(df[metric_name], errors="coerce")
    df = df.dropna(subset=[metric_name])
    if df.empty:
        return {}
    if metric_name == "loss":
        row = df.loc[df[metric_name].idxmin()]
    else:
        row = df.loc[df[metric_name].idxmax()]
    out: dict[str, Any] = {
        "metric": metric_name,
        "value": float(row[metric_name]),
    }
    for key in ["step", "epoch", "eval_kind"]:
        if key in row.index and pd.notna(row[key]):
            value = row[key]
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            out[key] = value
    return out


def _health_checks(train_df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if train_df.empty:
        return warnings

    if "loss" in train_df.columns:
        ordered = train_df
        if "__x_step" in train_df.columns:
            ordered = train_df.sort_values("__x_step")
        loss = pd.to_numeric(ordered["loss"], errors="coerce")
        if (~np.isfinite(loss.dropna())).any():
            warnings.append("loss contains NaN/Inf values")
        head = loss.dropna().iloc[:200]
        if len(head) >= 40:
            first = head.iloc[:20].mean()
            last = head.iloc[-20:].mean()
            if first > 0 and last > first * 1.5:
                warnings.append("early divergence: loss increases within first 200 steps")

    if "grad_norm" in train_df.columns:
        grad_norm = pd.to_numeric(train_df["grad_norm"], errors="coerce")
        if (grad_norm.dropna() > 1000).any():
            warnings.append("grad_norm spikes > 1000")

    return warnings


def analyze_training(
    train_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    eval_events_df: pd.DataFrame | None,
    outdir: Path,
    cfg: dict[str, Any],
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    plots_enabled = cfg.get("outputs", {}).get("write_plots", True)
    plots_dir = outdir / "plots"
    if plots_enabled:
        plots_dir.mkdir(parents=True, exist_ok=True)

    rolling = int(cfg.get("analysis", {}).get("rolling_window", 10))
    plot_cfg = cfg.get("plots", {})
    plot_style = cfg.get("plotting", {})
    cmap = plot_style.get("colormap")
    public_only = bool(plot_cfg.get("public_only", False))
    include_debug_plots = bool(plot_cfg.get("include_debug_plots", True))
    make_pretty = bool(plot_cfg.get("make_pretty", False))

    def _x_axis(df: pd.DataFrame) -> str | None:
        # global_step is the canonical x-axis; prefer explicit step.
        if "step" in df.columns:
            series = pd.to_numeric(df["step"], errors="coerce")
            if series.notna().any():
                return "step"
        if "__x_step" in df.columns:
            series = pd.to_numeric(df["__x_step"], errors="coerce")
            if series.notna().any():
                return "__x_step"
        return None

    x_train = _x_axis(train_df)
    x_metrics = _x_axis(metrics_df)

    train_plot_df = train_df
    if "split" in train_plot_df.columns:
        train_plot_df = train_plot_df[train_plot_df["split"] == "train"]
    if x_train and plots_enabled:
        if "lr" in train_plot_df.columns:
            plot_pretty_line(
                train_plot_df,
                x_train,
                "lr",
                group_by=None,
                rolling=rolling,
                outpath=plots_dir / "train_lr_vs_step.png",
                x_label="global_step",
                y_label="lr",
                title="Learning Rate Schedule",
                cmap=cmap,
            ) if make_pretty else plot_line(
                train_plot_df,
                x_train,
                "lr",
                group_by=None,
                rolling=rolling,
                outpath=plots_dir / "train_lr_vs_step.png",
                x_label="global_step",
                y_label="lr",
                title="Learning Rate Schedule",
            )
        if "tokens_per_s" in train_plot_df.columns:
            plot_pretty_line(
                train_plot_df,
                x_train,
                "tokens_per_s",
                group_by=None,
                rolling=rolling,
                outpath=plots_dir / "train_tokens_per_s_vs_step.png",
                x_label="global_step",
                y_label="tokens_per_s",
                title="Tokens per Second",
                cmap=cmap,
            ) if make_pretty else plot_line(
                train_plot_df,
                x_train,
                "tokens_per_s",
                group_by=None,
                rolling=rolling,
                outpath=plots_dir / "train_tokens_per_s_vs_step.png",
                x_label="global_step",
                y_label="tokens_per_s",
                title="Tokens per Second",
            )
        if include_debug_plots and not public_only:
            for y in ["loss", "grad_norm", "step_time_s"]:
                plot_line(
                    train_plot_df,
                    x_train,
                    y,
                    group_by="split",
                    rolling=rolling,
                    outpath=plots_dir / f"train_{y}_vs_step.png",
                    x_label="global_step",
                )

    if x_metrics and plots_enabled and include_debug_plots and plot_cfg.get("legacy_metrics_jsonl_plots", False):
        eval_metrics_df = _select_eval_split(metrics_df, ["eval_log", "eval_stop", "eval", "val", "validation", "dev"])
        for y in ["macro_f1", "acc", "f1_negative", "f1_neutral", "f1_positive"]:
            plot_line(
                eval_metrics_df,
                x_metrics,
                y,
                group_by="split",
                rolling=rolling,
                outpath=plots_dir / f"metrics_{y}_vs_step.png",
                x_label="global_step",
            )

    eval_events_df = eval_events_df if eval_events_df is not None else pd.DataFrame()
    eval_events_df = _expand_eval_metrics(eval_events_df)
    if "eval_kind" not in eval_events_df.columns:
        eval_events_df["eval_kind"] = None

    eval_log_df = eval_events_df[eval_events_df["eval_kind"] == "log"].copy()
    eval_stop_df = eval_events_df[eval_events_df["eval_kind"] == "stop"].copy()
    warnings = _health_checks(train_df)
    if eval_log_df.empty and eval_stop_df.empty:
        # train.jsonl is the authoritative eval stream; fall back to metrics.jsonl only if missing.
        warnings.append("eval events missing in train.jsonl; falling back to metrics.jsonl for eval plots")
        eval_log_df = _select_eval_split(metrics_df, ["eval_log"]).copy()
        eval_stop_df = _select_eval_split(metrics_df, ["eval_stop", "eval", "val", "validation", "dev"]).copy()
        for df in (eval_log_df, eval_stop_df):
            if "eval_loss" not in df.columns and "loss" in df.columns:
                df["eval_loss"] = df["loss"]

    if plots_enabled:
        def _series(df: pd.DataFrame, y_col: str, label: str) -> pd.DataFrame:
            x_col = _x_axis(df)
            if df.empty or x_col is None or y_col not in df.columns:
                return pd.DataFrame()
            out = df[[x_col, y_col]].copy()
            out = out.rename(columns={x_col: "step", y_col: "value"})
            out["series"] = label
            out["step"] = pd.to_numeric(out["step"], errors="coerce")
            out["value"] = pd.to_numeric(out["value"], errors="coerce")
            return out.dropna(subset=["step", "value"])

        # eval_stop is sparse; keep it on separate plots unless explicitly combined.
        combine_eval = plot_cfg.get("combine_eval_log_and_stop", False) and not public_only
        if plot_cfg.get("loss_train_vs_eval", True):
            loss_log_series = []
            if "loss" in train_plot_df.columns:
                loss_log_series.append(_series(train_plot_df, "loss", "train"))
            if "eval_loss" in eval_log_df.columns:
                loss_log_series.append(_series(eval_log_df, "eval_loss", "eval_log"))
            if loss_log_series:
                loss_log_plot_df = pd.concat(loss_log_series, ignore_index=True)
                (plot_pretty_line if make_pretty else plot_line)(
                    loss_log_plot_df,
                    "step",
                    "value",
                    group_by="series",
                    rolling=rolling,
                    outpath=plots_dir / "loss__train_vs_eval_log.png",
                    x_label="global_step",
                    y_label="loss",
                    title="Train vs Eval(Log) Loss",
                    cmap=cmap,
                )

            if "eval_loss" in eval_stop_df.columns:
                loss_stop_plot_df = _series(eval_stop_df, "eval_loss", "eval_stop")
                if not loss_stop_plot_df.empty:
                    (plot_pretty_line if make_pretty else plot_line)(
                        loss_stop_plot_df,
                        "step",
                        "value",
                        group_by=None,
                        rolling=1,
                        outpath=plots_dir / "loss__eval_stop.png",
                        x_label="global_step",
                        y_label="loss",
                        title="Eval Stop Loss",
                        marker="o",
                        cmap=cmap,
                    )

        if combine_eval:
            if plot_cfg.get("macro_f1_eval_log_vs_stop", True):
                f1_series = []
                if "macro_f1" in eval_log_df.columns:
                    f1_series.append(_series(eval_log_df, "macro_f1", "eval_log"))
                if "macro_f1" in eval_stop_df.columns:
                    f1_series.append(_series(eval_stop_df, "macro_f1", "eval_stop"))
                if f1_series:
                    f1_plot_df = pd.concat(f1_series, ignore_index=True)
                    plot_line(
                        f1_plot_df,
                        "step",
                        "value",
                        group_by="series",
                        rolling=1,
                        outpath=plots_dir / "macro_f1__eval_log_vs_stop.png",
                        x_label="global_step",
                        y_label="macro_f1",
                        title="Macro F1 (eval_log vs eval_stop)",
                        marker="o",
                    )

            if plot_cfg.get("acc_eval_log_vs_stop", False):
                acc_series = []
                if "acc" in eval_log_df.columns:
                    acc_series.append(_series(eval_log_df, "acc", "eval_log"))
                if "acc" in eval_stop_df.columns:
                    acc_series.append(_series(eval_stop_df, "acc", "eval_stop"))
                if acc_series:
                    acc_plot_df = pd.concat(acc_series, ignore_index=True)
                    plot_line(
                        acc_plot_df,
                        "step",
                        "value",
                        group_by="series",
                        rolling=1,
                        outpath=plots_dir / "acc__eval_log_vs_stop.png",
                        x_label="global_step",
                        y_label="acc",
                        title="Accuracy (eval_log vs eval_stop)",
                        marker="o",
                    )
        else:
            if "macro_f1" in eval_log_df.columns:
                f1_log_plot_df = _series(eval_log_df, "macro_f1", "eval_log")
                if not f1_log_plot_df.empty:
                    (plot_pretty_line if make_pretty else plot_line)(
                        f1_log_plot_df,
                        "step",
                        "value",
                        group_by=None,
                        rolling=1,
                        outpath=plots_dir / "macro_f1__eval_log.png",
                        x_label="global_step",
                        y_label="macro_f1",
                        title="Macro F1 (eval_log)",
                        marker="o",
                        cmap=cmap,
                    )
            if "macro_f1" in eval_stop_df.columns:
                f1_stop_plot_df = _series(eval_stop_df, "macro_f1", "eval_stop")
                if not f1_stop_plot_df.empty:
                    (plot_pretty_line if make_pretty else plot_line)(
                        f1_stop_plot_df,
                        "step",
                        "value",
                        group_by=None,
                        rolling=1,
                        outpath=plots_dir / "macro_f1__eval_stop.png",
                        x_label="global_step",
                        y_label="macro_f1",
                        title="Macro F1 (eval_stop)",
                        marker="o",
                        cmap=cmap,
                    )

            if plot_cfg.get("acc_eval_log_vs_stop", False):
                if "acc" in eval_log_df.columns:
                    acc_log_plot_df = _series(eval_log_df, "acc", "eval_log")
                    if not acc_log_plot_df.empty:
                        (plot_pretty_line if make_pretty else plot_line)(
                            acc_log_plot_df,
                            "step",
                            "value",
                            group_by=None,
                            rolling=1,
                            outpath=plots_dir / "acc__eval_log.png",
                            x_label="global_step",
                            y_label="acc",
                            title="Accuracy (eval_log)",
                            marker="o",
                            cmap=cmap,
                        )
                if "acc" in eval_stop_df.columns:
                    acc_stop_plot_df = _series(eval_stop_df, "acc", "eval_stop")
                    if not acc_stop_plot_df.empty:
                        (plot_pretty_line if make_pretty else plot_line)(
                            acc_stop_plot_df,
                            "step",
                            "value",
                            group_by=None,
                            rolling=1,
                            outpath=plots_dir / "acc__eval_stop.png",
                            x_label="global_step",
                            y_label="acc",
                            title="Accuracy (eval_stop)",
                            marker="o",
                            cmap=cmap,
                        )

        if plot_cfg.get("per_class_f1_plots", False):
            for metric in ["f1_negative", "f1_neutral", "f1_positive"]:
                if combine_eval:
                    series = []
                    if metric in eval_log_df.columns:
                        series.append(_series(eval_log_df, metric, "eval_log"))
                    if metric in eval_stop_df.columns:
                        series.append(_series(eval_stop_df, metric, "eval_stop"))
                    if not series:
                        continue
                    plot_df = pd.concat(series, ignore_index=True)
                    plot_line(
                        plot_df,
                        "step",
                        "value",
                        group_by="series",
                        rolling=1,
                        outpath=plots_dir / f"{metric}__eval_log_vs_stop.png",
                        x_label="global_step",
                        y_label=metric,
                        title=f"{metric} (eval_log vs eval_stop)",
                        marker="o",
                    )
                else:
                    if metric in eval_log_df.columns:
                        log_df = _series(eval_log_df, metric, "eval_log")
                        if not log_df.empty:
                            (plot_pretty_line if make_pretty else plot_line)(
                                log_df,
                                "step",
                                "value",
                                group_by=None,
                                rolling=1,
                                outpath=plots_dir / f"{metric}__eval_log.png",
                                x_label="global_step",
                                y_label=metric,
                                title=f"{metric} (eval_log)",
                                marker="o",
                                cmap=cmap,
                            )
                    if metric in eval_stop_df.columns:
                        stop_df = _series(eval_stop_df, metric, "eval_stop")
                        if not stop_df.empty:
                            (plot_pretty_line if make_pretty else plot_line)(
                                stop_df,
                                "step",
                                "value",
                                group_by=None,
                                rolling=1,
                                outpath=plots_dir / f"{metric}__eval_stop.png",
                                x_label="global_step",
                                y_label=metric,
                                title=f"{metric} (eval_stop)",
                                marker="o",
                                cmap=cmap,
                            )

    best = compute_best(metrics_df, cfg)

    cm_last = None
    if "confusion_matrix" in eval_stop_df.columns:
        cm_series = eval_stop_df["confusion_matrix"].dropna()
        if len(cm_series):
            cm_last = cm_series.iloc[-1]
    if cm_last is None and "confusion_matrix" in metrics_df.columns:
        cm_series = metrics_df["confusion_matrix"].dropna()
        if len(cm_series):
            cm_last = cm_series.iloc[-1]

    if cm_last is not None and plots_enabled:
        plot_confusion_matrix(
            cm_last,
            plots_dir / "confusion_matrix_last.png",
            "confusion_matrix (last)",
            labels=labels,
            cmap=cmap,
        )

    if best and "confusion_matrix" in metrics_df.columns and plots_enabled:
        if "__x_step" in metrics_df.columns and "x_step" in best:
            hit = metrics_df[metrics_df["__x_step"] == best["x_step"]]
        elif "step" in metrics_df.columns and "step" in best:
            step_val = best["step"]
            try:
                step_val = float(step_val)
            except Exception:
                pass
            hit = metrics_df[metrics_df["step"] == step_val]
        else:
            hit = pd.DataFrame()
        if not hit.empty:
            cm = hit.iloc[0].get("confusion_matrix")
            if isinstance(cm, list):
                plot_confusion_matrix(cm, plots_dir / "confusion_matrix_best.png", "confusion_matrix (best)")

    # Stop-eval is preferred for best selection because it reflects the early-stopping cadence.
    best_by_stop_eval = compute_best_from_eval_events(eval_stop_df, cfg)
    if not best_by_stop_eval:
        # Stop-eval is preferred for best selection; fall back to metrics.jsonl for older runs.
        warnings.append("eval_stop events missing; falling back to metrics.jsonl for best selection")
        best_by_stop_eval = best

    if not eval_stop_df.empty and not metrics_df.empty:
        metrics_eval = _select_eval_split(metrics_df, ["eval_stop", "eval", "val", "validation", "dev"])
        if "step" in eval_stop_df.columns and "step" in metrics_eval.columns:
            eval_steps = pd.to_numeric(eval_stop_df["step"], errors="coerce").dropna().unique()
            metrics_steps = pd.to_numeric(metrics_eval["step"], errors="coerce").dropna().unique()
            if len(eval_steps) != len(metrics_steps) or set(eval_steps) != set(metrics_steps):
                warnings.append("eval_stop steps do not match metrics.jsonl steps")

    summary = {
        "rows": {
            "train": int(len(train_df)),
            "metrics": int(len(metrics_df)),
            "eval_events": int(len(eval_events_df)),
        },
        "best": best,
        "best_by_stop_eval": best_by_stop_eval,
        "log_eval_count": int(len(eval_log_df)),
        "stop_eval_count": int(len(eval_stop_df)),
        "health_warnings": warnings,
    }
    return summary
