from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=max(1, window // 4)).mean()


def plot_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    group_by: str | None,
    rolling: int,
    outpath: Path,
    *,
    x_label: str | None = None,
    y_label: str | None = None,
    marker: str | None = None,
    title: str | None = None,
) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return

    plt.figure()
    if group_by and group_by in df.columns:
        for g, dfg in df.groupby(group_by):
            if y not in dfg.columns:
                continue
            ys_raw = pd.to_numeric(dfg[y], errors="coerce")
            if ys_raw.notna().sum() == 0:
                continue
            dfg = dfg.sort_values(x)
            ys = _rolling_mean(pd.to_numeric(dfg[y], errors="coerce"), rolling)
            plt.plot(dfg[x], ys, label=str(g), marker=marker)
        plt.legend()
    else:
        dfg = df.sort_values(x)
        ys = _rolling_mean(pd.to_numeric(dfg[y], errors="coerce"), rolling)
        if ys.notna().sum() == 0:
            plt.close()
            return
        plt.plot(dfg[x], ys, marker=marker)

    if title:
        plt.title(title)
    plt.xlabel(x_label if x_label is not None else x)
    plt.ylabel(y_label if y_label is not None else y)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_confusion_matrix(
    cm: Any,
    outpath: Path,
    title: str,
    *,
    labels: Iterable[str] | None = None,
    cmap: str | None = None,
) -> None:
    if not isinstance(cm, list) or not cm or not isinstance(cm[0], list):
        return
    mat = np.array(cm, dtype=float)
    label_list = list(labels) if labels is not None else None
    if label_list is not None and len(label_list) != mat.shape[0]:
        label_list = None
    plt.figure()
    try:
        import seaborn as sns

        sns.set_theme(context="notebook", style="whitegrid")
        sns.heatmap(
            mat,
            annot=True,
            fmt=".0f",
            cbar=True,
            xticklabels=label_list,
            yticklabels=label_list,
            cmap=cmap or "viridis",
        )
        plt.title(title, fontsize=12)
        plt.xlabel("pred", fontsize=10)
        plt.ylabel("true", fontsize=10)
    except Exception:
        plt.imshow(mat)
        plt.title(title)
        plt.xlabel("pred")
        plt.ylabel("true")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_pretty_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    group_by: str | None,
    rolling: int,
    outpath: Path,
    *,
    x_label: str,
    y_label: str,
    title: str,
    marker: str | None = None,
    cmap: str | None = None,
) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return
    df = df.copy()
    df[x] = pd.to_numeric(df[x], errors="coerce")
    df[y] = pd.to_numeric(df[y], errors="coerce")
    df = df.dropna(subset=[x, y])
    if df.empty:
        return
    try:
        import seaborn as sns

        # Use a perceptually uniform, colorblind-safe colormap for publication.
        sns.set_theme(context="notebook", style="whitegrid")
        if cmap:
            plt.set_cmap(cmap)
        plt.figure()
        if group_by and group_by in df.columns:
            for name, dfg in df.groupby(group_by):
                dfg = dfg.sort_values(x)
                ys = _rolling_mean(dfg[y], rolling)
                sns.lineplot(x=dfg[x], y=ys, label=str(name), marker=marker)
            plt.legend()
        else:
            dfg = df.sort_values(x)
            ys = _rolling_mean(dfg[y], rolling)
            sns.lineplot(x=dfg[x], y=ys, marker=marker)
        plt.title(title, fontsize=12)
        plt.xlabel(x_label, fontsize=10)
        plt.ylabel(y_label, fontsize=10)
        plt.tight_layout()
        plt.savefig(outpath, dpi=160)
        plt.close()
    except Exception:
        plot_line(
            df,
            x,
            y,
            group_by=group_by,
            rolling=rolling,
            outpath=outpath,
            x_label=x_label,
            y_label=y_label,
            marker=marker,
            title=title,
        )
