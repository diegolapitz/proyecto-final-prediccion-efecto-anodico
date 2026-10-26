"""Proyecto final UTN: preparación, entrenamiento y evaluación temporal de EA.

Ejemplo sin datos privados: python proyecto_final_ea.py --demo
El modo demostración genera datos artificiales; no replica los resultados reales.
El modo privado recibe dos CSV normalizados: mediciones y episodios de EA.
Las 42 variables usan únicamente la observación actual y su historia disponible.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import tempfile
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260825
HORIZON = 60
NEGATIVE_FRACTION = 0.02
MINUTE = 60_000_000_000
TRAIN_START = pd.Timestamp("2026-01-01")
VALID_START = pd.Timestamp("2026-06-01")
TEST_START = pd.Timestamp("2026-07-01")
TEST_END = pd.Timestamp("2026-08-01")
LABELS = {"logistic_regression": "Regresión Logística",
          "lightgbm": "LightGBM", "extra_trees": "ExtraTrees"}


# 1. Entrada normalizada y demostración independiente de los datos industriales.
def synthetic_data():
    """Cuatro unidades ficticias, cadencia de cinco minutos y eventos simulados."""
    rng = np.random.default_rng(SEED)
    times = pd.date_range(TRAIN_START, TEST_END, freq="5min", inclusive="left")
    frames, events = [], []
    for unit in range(4):
        unit_id = f"demo_{unit + 1}"
        indices = np.arange(400 + unit * 43, len(times) - 20, 620)
        risk = np.zeros(len(times))
        for j, index in enumerate(indices):
            risk[index - 12:index] = np.linspace(0, 1, 12)
            start = times[index] + pd.Timedelta(seconds=130)
            events.append({"event_id": f"synthetic_{unit}_{j}", "unit_id": unit_id,
                           "start": start, "end": start + pd.Timedelta(seconds=57),
                           "observed_at": start + pd.Timedelta(seconds=170)})
        frames.append(pd.DataFrame({
            "timestamp": times, "unit_id": unit_id,
            "current": 100 + rng.normal(0, 1, len(times)),
            "voltage": 4 + 0.15 * risk + rng.normal(0, 0.08, len(times)),
            "resistance": 40 + 2 * risk + rng.normal(0, 0.7, len(times)),
            "is_shutdown": np.zeros(len(times), dtype=int)}))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(events)


def load_inputs(con, args):
    if args.demo:
        measurements, episodes = synthetic_data()
        con.register("input_measurements", measurements)
        con.register("input_events", episodes)
    else:
        for table, path in [("input_measurements", args.measurements),
                            ("input_events", args.events)]:
            # Lectura textual para conservar los identificadores y exigir tipos explícitos.
            con.read_csv(str(path), all_varchar=True).create_view(table)
    con.execute("""
        CREATE TABLE measurements AS SELECT
            CAST(timestamp AS TIMESTAMP_NS) AS timestamp, CAST(unit_id AS VARCHAR) AS unit_id,
            CASE WHEN isfinite(CAST(current AS DOUBLE)) THEN CAST(current AS DOUBLE) END AS current,
            CASE WHEN isfinite(CAST(voltage AS DOUBLE)) THEN CAST(voltage AS DOUBLE) END AS voltage,
            CASE WHEN isfinite(CAST(resistance AS DOUBLE)) THEN CAST(resistance AS DOUBLE) END AS resistance,
            CAST(is_shutdown AS INTEGER) AS is_shutdown
        FROM input_measurements;
        CREATE TABLE events AS SELECT CAST(event_id AS VARCHAR) AS event_id,
            CAST(unit_id AS VARCHAR) AS unit_id, CAST(start AS TIMESTAMP_NS) AS start,
            CAST("end" AS TIMESTAMP_NS) AS "end",
            CAST(observed_at AS TIMESTAMP_NS) AS observed_at FROM input_events;
    """)
    invalid = con.execute("""
        SELECT
          (SELECT count(*) FROM measurements WHERE timestamp IS NULL OR unit_id IS NULL
             OR unit_id='' OR is_shutdown IS NULL OR is_shutdown NOT IN (0,1))
          + (SELECT count(*) FROM events WHERE event_id IS NULL OR event_id='' OR unit_id IS NULL
             OR unit_id='' OR start IS NULL OR "end" IS NULL OR observed_at IS NULL
             OR "end" < start OR observed_at < start)
          + (SELECT count(*) FROM (SELECT unit_id,timestamp FROM measurements
             GROUP BY ALL HAVING count(*)>1))
          + (SELECT count(*) FROM (SELECT event_id FROM events GROUP BY ALL HAVING count(*)>1))
          + (SELECT count(*) FROM (SELECT unit_id,start FROM events GROUP BY ALL HAVING count(*)>1))
          + (SELECT count(*) FROM (SELECT unit_id,observed_at FROM events GROUP BY ALL HAVING count(*)>1))
    """).fetchone()[0]
    if invalid:
        raise ValueError("Entrada inválida: revisar claves, duplicados, fechas y estado de parada.")
    print("Entradas validadas.", flush=True)


# 2. Variables temporales: ventanas inclusivas [t-ventana, t], nunca futuras.
def feature_names():
    names = ["current", "voltage", "resistance"]
    for minutes in (15, 30, 60, 120, 480):
        names += [f"resistance_{stat}_{minutes}m" for stat in ("mean", "std", "range")]
    names += ["resistance_min_120m", "voltage_mean_30m", "voltage_max_30m",
              "voltage_mean_60m", "voltage_max_60m", "current_mean_60m",
              "current_std_60m", "max_gap_120m_s"]
    for minutes in (15, 30, 60):
        names += [f"lag_age_{minutes}m_s", f"resistance_diff_{minutes}m", f"voltage_diff_{minutes}m"]
    return names + ["resistance_distance_mean_60m", "resistance_z_60m",
                    "resistance_distance_min_120m", "hour_sin", "hour_cos",
                    "weekday_sin", "weekday_cos"]


def feature_query(start, end):
    """Una partición mensual con 24 horas de contexto para los bordes de ventana."""
    window = "PARTITION BY unit_id ORDER BY timestamp"
    lags = [f"lag(timestamp) OVER ({window}) AS previous_timestamp"]
    differences = []
    for minutes, periods in ((15, 3), (30, 6), (60, 12)):
        lags.append(f"lag(timestamp,{periods}) OVER ({window}) AS lag_time_{minutes}")
        differences.append(f"date_diff('second',lag_time_{minutes},timestamp) AS lag_age_{minutes}m_s")
        for signal in ("resistance", "voltage"):
            lags.append(f"lag({signal},{periods}) OVER ({window}) AS {signal}_lag_{minutes}")
            differences.append(f"{signal}-{signal}_lag_{minutes} AS {signal}_diff_{minutes}m")
    rolls = []
    for minutes in (15, 30, 60, 120, 480):
        rolls += [f"avg(resistance) OVER w{minutes} AS resistance_mean_{minutes}m",
                  f"stddev_samp(resistance) OVER w{minutes} AS resistance_std_{minutes}m",
                  f"max(resistance) OVER w{minutes}-min(resistance) OVER w{minutes} AS resistance_range_{minutes}m"]
    rolls += ["min(resistance) OVER w120 AS resistance_min_120m",
              "avg(voltage) OVER w30 AS voltage_mean_30m", "max(voltage) OVER w30 AS voltage_max_30m",
              "avg(voltage) OVER w60 AS voltage_mean_60m", "max(voltage) OVER w60 AS voltage_max_60m",
              "avg(current) OVER w60 AS current_mean_60m", "stddev_samp(current) OVER w60 AS current_std_60m",
              "max(gap_s) OVER w120 AS max_gap_120m_s"]
    windows = ",".join(f"w{m} AS ({window} RANGE BETWEEN INTERVAL {m} MINUTE PRECEDING AND CURRENT ROW)"
                       for m in (15, 30, 60, 120, 480))
    # Los eventos se usan sólo para etiquetas/elegibilidad: nunca como predictores.
    return f"""
        WITH raw AS (
            SELECT * FROM measurements WHERE timestamp >= TIMESTAMP '{start}' - INTERVAL 24 HOUR
              AND timestamp < TIMESTAMP '{end}'
        ), lagged AS (SELECT *, {','.join(lags)} FROM raw),
        gaps AS (SELECT *, date_diff('second',previous_timestamp,timestamp) AS gap_s FROM lagged),
        rolled AS (SELECT *, {','.join(rolls)} FROM gaps WINDOW {windows}),
        derived AS (SELECT *, {','.join(differences)},
            resistance-resistance_mean_60m AS resistance_distance_mean_60m,
            CASE WHEN resistance_std_60m>0 THEN (resistance-resistance_mean_60m)/resistance_std_60m END AS resistance_z_60m,
            resistance-resistance_min_120m AS resistance_distance_min_120m,
            sin(2*pi()*extract('hour' FROM timestamp)/24.0) AS hour_sin,
            cos(2*pi()*extract('hour' FROM timestamp)/24.0) AS hour_cos,
            sin(2*pi()*extract('dow' FROM timestamp)/7.0) AS weekday_sin,
            cos(2*pi()*extract('dow' FROM timestamp)/7.0) AS weekday_cos FROM rolled)
        SELECT d.timestamp,d.unit_id,{','.join('d.'+n for n in feature_names())},
            (next_event.start <= d.timestamp + INTERVAL {HORIZON} MINUTE) IS TRUE AS target,
            (past_event.start IS NULL OR d.timestamp > past_event."end" + INTERVAL 30 MINUTE)
            AND coalesce(d.max_gap_120m_s,d.gap_s,0)<=1800 AND d.is_shutdown=0
            AND d.current IS NOT NULL AND d.voltage IS NOT NULL AND d.resistance IS NOT NULL AS eligible
        FROM derived d
        ASOF LEFT JOIN events past_event ON d.unit_id=past_event.unit_id AND d.timestamp>=past_event.observed_at
        ASOF LEFT JOIN events next_event ON d.unit_id=next_event.unit_id AND d.timestamp<next_event.start
        WHERE d.timestamp >= TIMESTAMP '{start}' AND d.timestamp < TIMESTAMP '{end}'
    """


def prepare_features(con):
    boundaries = pd.date_range(TRAIN_START, TEST_END, freq="MS")
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        prefix = "CREATE TABLE features AS " if i == 0 else "INSERT INTO features "
        con.execute(prefix + feature_query(start, end))
        print(f"Variables preparadas: {start:%Y-%m} (42 columnas).", flush=True)


# 3. Entrenamiento: enero-mayo; validación: junio; evaluación: julio.
def make_model(algorithm):
    if algorithm == "lightgbm":
        return LGBMClassifier(objective="binary", random_state=SEED, n_jobs=2,
                              verbosity=-1, n_estimators=350, learning_rate=0.05,
                              num_leaves=31, min_child_samples=100, subsample=0.85,
                              colsample_bytree=0.85)
    estimator = (LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED)
                 if algorithm == "logistic_regression" else
                 ExtraTreesClassifier(n_estimators=250, min_samples_leaf=10,
                                      max_features="sqrt", random_state=SEED, n_jobs=2))
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if algorithm == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps + [("model", estimator)])


def model_matrix(frame):
    return frame[feature_names()].replace([np.inf, -np.inf], np.nan).astype(np.float32)


def row_metrics(y, score, threshold):
    if len(np.unique(y)) != 2:
        raise ValueError("Cada período debe contener positivos y negativos evaluables.")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, score >= threshold, average="binary", zero_division=0)
    return dict(n=len(y), positives=int(np.sum(y)), prevalence=float(np.mean(y)),
                pr_auc=average_precision_score(y, score), roc_auc=roc_auc_score(y, score),
                brier=brier_score_loss(y, score), threshold=float(threshold),
                precision=precision, recall=recall, f1=f1)


def score_period(con, model, start, end):
    # Se puntúan bloques para evitar cargar todas las variables del período en RAM.
    cursor = con.execute("SELECT * FROM features WHERE timestamp>=? AND timestamp<? ORDER BY timestamp,unit_id",
                         [start, end])
    parts = []
    while True:
        frame = cursor.fetch_df_chunk(40)
        if frame.empty:
            break
        scored = frame[["unit_id", "timestamp", "eligible", "target"]].copy()
        scored["score"] = model.predict_proba(model_matrix(frame))[:, 1].astype(np.float64)
        parts.append(scored)
    if not parts:
        raise ValueError("No hay mediciones en uno de los períodos requeridos.")
    return pd.concat(parts, ignore_index=True)


def train_and_select(con, work):
    # Positivos completos + 2 % de negativos, con corrección de peso por muestreo.
    train = con.execute(f"""SELECT {','.join(feature_names())},target FROM features
        WHERE eligible AND timestamp>=? AND timestamp<?
        AND (target OR hash(unit_id,timestamp,{SEED+HORIZON}) % 1000000 < 20000)
        ORDER BY timestamp,unit_id""", [TRAIN_START, VALID_START-pd.Timedelta(minutes=HORIZON)]).fetchdf()
    y = train.pop("target").to_numpy(dtype=np.int8)
    if len(np.unique(y)) != 2:
        raise ValueError("El entrenamiento necesita ejemplos de ambas clases.")
    x = model_matrix(train)
    weights = np.where(y == 1, 1.0, 1.0 / NEGATIVE_FRACTION)
    del train
    candidates = []
    for algorithm in LABELS:
        model = make_model(algorithm)
        kwargs = {"sample_weight" if algorithm == "lightgbm" else "model__sample_weight": weights}
        model.fit(x, y, **kwargs)
        validation = score_period(con, model, VALID_START, TEST_START-pd.Timedelta(minutes=HORIZON))
        validation = validation.loc[validation.eligible]
        precision, recall, thresholds = precision_recall_curve(validation.target, validation.score)
        f1 = 2*precision[:-1]*recall[:-1] / np.maximum(precision[:-1]+recall[:-1], 1e-15)
        threshold = float(thresholds[np.argmax(f1)])
        metrics = row_metrics(validation.target.to_numpy(), validation.score.to_numpy(), threshold)
        candidates.append({"algorithm": algorithm, "threshold": threshold, "validation": metrics})
        with (work / f"{algorithm}.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        print(f"Validación {LABELS[algorithm]}: AP={metrics['pr_auc']:.5f}", flush=True)
        del model, validation
        gc.collect()
    selected = max(candidates, key=lambda item: item["validation"]["pr_auc"])["algorithm"]
    return candidates, selected


# 4. Emisiones reales de alarma y evaluación por episodio (sin duplicar créditos).
def emit_alerts(predictions, threshold, cooldown_minutes):
    candidates = predictions.loc[predictions.eligible & (predictions.score >= threshold),
                                 ["unit_id", "timestamp", "score"]].copy()
    candidates = candidates.sort_values(["unit_id", "timestamp"], kind="stable")
    kept = []
    for _, group in candidates.groupby("unit_id", sort=False):
        last = None
        for index, timestamp in zip(group.index, group.timestamp.astype("int64")):
            if last is None or timestamp-last >= cooldown_minutes*MINUTE:
                kept.append(index)
                last = timestamp  # Última emisión, no último score alto.
    return candidates.loc[kept].reset_index(drop=True)


def evaluate_episodes(predictions, events, threshold, cooldown, start, end, outcome_end):
    p = predictions.loc[(predictions.timestamp >= start) & (predictions.timestamp < end)]
    events = events.copy()
    events["start"] = pd.to_datetime(events.start)
    events = events.loc[events.start <= outcome_end].copy()
    emitted = emit_alerts(p, threshold, cooldown)
    emitted["matched_event_id"] = pd.Series(index=emitted.index, dtype=object)
    emitted["lead_minutes"] = np.nan
    emitted["true_alert"] = False
    emitted["censored"] = emitted.timestamp + pd.Timedelta(minutes=HORIZON) > outcome_end
    groups = {key: group.sort_values("start") for key, group in events.groupby("unit_id")}
    for unit, group in emitted.groupby("unit_id", sort=False):
        e = groups.get(unit)
        if e is None or e.empty:
            continue
        ts, starts = group.timestamp.astype("int64").to_numpy(), e.start.astype("int64").to_numpy()
        next_index = np.searchsorted(starts, ts, side="right")
        locations = np.flatnonzero(next_index < len(starts))
        locations = locations[(starts[next_index[locations]]-ts[locations]) <= HORIZON*MINUTE]
        indices = group.index.to_numpy()[locations]
        emitted.loc[indices, "matched_event_id"] = e.iloc[next_index[locations]].event_id.to_numpy()
        emitted.loc[indices, "lead_minutes"] = (starts[next_index[locations]]-ts[locations])/MINUTE
        emitted.loc[indices, "true_alert"] = True
        emitted.loc[indices, "censored"] = False
    evaluated = events.loc[(events.start >= start) & (events.start < end)].copy()
    first = emitted.loc[emitted.true_alert].sort_values("timestamp").drop_duplicates("matched_event_id")
    first = first.set_index("matched_event_id")
    evaluated["lead_minutes"] = evaluated.event_id.map(first.lead_minutes)
    evaluated["detected"] = evaluated.lead_minutes.notna()
    opportunities = {}
    valid_groups = {unit: np.sort(group.timestamp.astype("int64").to_numpy())
                    for unit, group in p.loc[p.eligible].groupby("unit_id")}
    for unit, group in evaluated.groupby("unit_id"):
        ts = valid_groups.get(unit, np.empty(0, dtype="int64"))
        starts = group.start.astype("int64").to_numpy()
        counts = np.searchsorted(ts, starts, side="left")-np.searchsorted(ts, starts-HORIZON*MINUTE, side="left")
        opportunities.update(zip(group.event_id, counts))
    evaluated["opportunity_rows"] = evaluated.event_id.map(opportunities)
    n, detected = len(evaluated), int(evaluated.detected.sum())
    leads = evaluated.loc[evaluated.detected, "lead_minutes"]
    days = (end-start).total_seconds()/86400
    false = int((~emitted.true_alert & ~emitted.censored).sum())
    classified = int((~emitted.censored).sum())
    metrics = dict(threshold=threshold, cooldown_minutes=cooldown, events=n,
                   detected_events=detected, missed_events=n-detected,
                   episode_recall=detected/n if n else np.nan,
                   events_without_opportunity=int((evaluated.opportunity_rows == 0).sum()),
                   lead_median_min=leads.median(), lead_mean_min=leads.mean(),
                   lead_p25_min=leads.quantile(.25), lead_p75_min=leads.quantile(.75),
                   calendar_days=days, unique_alerts=len(emitted), alerts_per_day=len(emitted)/days,
                   false_alerts=false, false_alerts_per_day=false/days,
                   true_alerts=int(emitted.true_alert.sum()), outcome_censored_alerts=int(emitted.censored.sum()),
                   alert_precision=float(emitted.true_alert.sum()/classified) if classified else np.nan,
                   repeated_true_alerts_same_ea=int(emitted.true_alert.sum()-emitted.loc[emitted.true_alert,"matched_event_id"].nunique()))
    for minutes in (10, 20, 30):
        metrics[f"fraction_all_events_lead_ge_{minutes}m"] = float((leads >= minutes).sum()/n) if n else np.nan
        metrics[f"fraction_detected_lead_ge_{minutes}m"] = float((leads >= minutes).mean()) if detected else np.nan
    return metrics, evaluated


# 5. Cuatro figuras y tres tablas agregadas; nunca se exportan filas industriales.
def make_figures(rows, operations, selected_predictions, selected_events, output, demo):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    teal, orange, gray = "#0B6574", "#C86A24", "#83959D"
    chosen = rows.loc[rows.selected].iloc[0]
    op = operations.loc[operations.selected & operations.cooldown_minutes.eq(30)].iloc[0]
    note = "Demostración sintética" if demo else "Julio 2026 · evaluación retrospectiva"
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / name, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    comparison = rows.sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(comparison.algorithm.map(LABELS), comparison.pr_auc,
                   color=[teal if item else gray for item in comparison.selected])
    for bar, value in zip(bars, comparison.pr_auc):
        ax.text(value + comparison.pr_auc.max()*.025, bar.get_y()+bar.get_height()/2,
                f"{value:.4f}", va="center")
    ax.axvline(chosen.prevalence, color=orange, ls="--", label="Prevalencia")
    ax.set(xlim=(0, max(comparison.pr_auc.max()*1.25, .001)), xlabel="Average precision (PR-AUC)",
           title=f"Comparación a 60 minutos · 42 variables\n{note}")
    ax.legend(frameon=False)
    save(fig, "comparacion_modelos.png")

    valid = selected_predictions.loc[selected_predictions.eligible]
    precision, recall, _ = precision_recall_curve(valid.target, valid.score)
    stride = max(1, len(precision)//5000)
    indices = np.unique(np.r_[np.arange(0, len(precision), stride), len(precision)-1])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(recall[indices], precision[indices], color=teal, label=LABELS[chosen.algorithm])
    ax.scatter([chosen.recall], [chosen.precision], color=orange, label="Umbral elegido en junio", zorder=3)
    ax.axhline(chosen.prevalence, color=gray, ls="--", label="Prevalencia")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall por fila", ylabel="Precision por fila",
           title=f"Curva Precision–Recall · 60 minutos\n{note}")
    ax.legend(frameon=False)
    save(fig, "precision_recall.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.8))
    for ax, value, label, color in zip(axes,
            [100*op.episode_recall, op.alerts_per_day, op.false_alerts_per_day],
            ["EA detectados (%)", "Alertas / día", "Falsas alertas / día"], [teal, gray, orange]):
        ax.bar([label], [value], color=color, width=.5)
        ax.text(0, value+max(value*.04, .02), f"{value:.2f}", ha="center", fontweight="bold")
        ax.set_ylim(0, max(value*1.3, 1))
    fig.suptitle(f"{LABELS[chosen.algorithm]} · cooldown 30 min · escalas independientes\n{note}")
    save(fig, "metricas_operativas.png")

    leads = selected_events.loc[selected_events.detected, "lead_minutes"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(leads, bins=np.arange(0, 65, 5), color=teal, edgecolor="white")
    if len(leads):
        ax.axvline(leads.median(), color=orange, label=f"Mediana: {leads.median():.2f} min")
        ax.legend(frameon=False)
    ax.set(xlim=(0, 60), xlabel="Minutos desde la primera alerta válida al inicio estimado",
           ylabel="Episodios detectados", title=f"Anticipación por episodio de EA\n{note}")
    save(fig, "lead_time.png")


def evaluate_test(con, work, candidates, selected, output, outcome_end, demo):
    events = con.execute("SELECT * FROM events WHERE start>=? AND start<=?",
                         [TEST_START, outcome_end]).fetchdf()
    rows, operations = [], []
    selected_predictions = selected_events = None
    for item in candidates:
        algorithm, threshold = item["algorithm"], item["threshold"]
        # Sólo se abre el archivo temporal que acaba de producir este proceso.
        with (work / f"{algorithm}.pkl").open("rb") as handle:
            model = pickle.load(handle)
        predictions = score_period(con, model, TEST_START, TEST_END)
        valid = predictions.loc[predictions.eligible]
        base = dict(algorithm=algorithm, horizon_minutes=HORIZON, n_features=42,
                    selected=algorithm == selected,
                    split="synthetic_demo" if demo else "test_july_retrospective")
        rows.append({**base, **row_metrics(valid.target.to_numpy(), valid.score.to_numpy(), threshold)})
        for cooldown in (0, 30):
            metrics, details = evaluate_episodes(predictions, events, threshold, cooldown,
                                                 TEST_START, TEST_END, outcome_end)
            operations.append({**base, **metrics})
            if algorithm == selected and cooldown == 30:
                selected_predictions, selected_events = predictions, details
        del model
        gc.collect()
    rows, operations = pd.DataFrame(rows), pd.DataFrame(operations)
    folder = output / "results"
    folder.mkdir(parents=True, exist_ok=True)
    rows.to_csv(folder / "comparacion_modelos.csv", index=False)
    common = ["algorithm", "horizon_minutes", "n_features", "selected", "split", "threshold", "cooldown_minutes"]
    episode_columns = ["events", "detected_events", "missed_events", "episode_recall",
                       "events_without_opportunity", "lead_median_min", "lead_mean_min", "lead_p25_min", "lead_p75_min"]
    episode_columns += [n for n in operations if n.startswith("fraction_")]
    alert_columns = ["calendar_days", "unique_alerts", "alerts_per_day", "false_alerts", "false_alerts_per_day",
                     "true_alerts", "outcome_censored_alerts", "alert_precision", "repeated_true_alerts_same_ea"]
    operations[common+episode_columns].to_csv(folder / "metricas_por_episodio.csv", index=False)
    operations[common+alert_columns].to_csv(folder / "metricas_alertas.csv", index=False)
    make_figures(rows, operations, selected_predictions, selected_events, output, demo)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="Ejecutar con datos sintéticos.")
    parser.add_argument("--measurements", type=Path, help="CSV privado con mediciones normalizadas.")
    parser.add_argument("--events", type=Path, help="CSV privado con episodios normalizados.")
    parser.add_argument("--outcome-end", help="Fin confirmado de cobertura de episodios, formato YYYY-MM-DD HH:MM:SS.")
    parser.add_argument("--output", type=Path, help="Carpeta de salida (por defecto outputs/demo o outputs/local).")
    args = parser.parse_args()
    if args.demo and (args.measurements or args.events or args.outcome_end):
        parser.error("Usar --demo sin archivos privados ni --outcome-end.")
    if not args.demo and not (args.measurements and args.events and args.outcome_end):
        parser.error("Indicar --demo o bien --measurements, --events y --outcome-end.")
    outcome_end = TEST_END+pd.Timedelta(minutes=HORIZON) if args.demo else pd.Timestamp(args.outcome_end)
    if outcome_end.tzinfo is not None or outcome_end < TEST_END+pd.Timedelta(minutes=HORIZON):
        parser.error("Se requiere cobertura hasta 2026-08-01 01:00:00, con fechas sin zona horaria.")
    # El seguimiento se limita al necesario para la última predicción de julio.
    outcome_end = TEST_END+pd.Timedelta(minutes=HORIZON)
    output = (args.output or Path("outputs") / ("demo" if args.demo else "local")).resolve()
    root = Path(__file__).resolve().parent
    protected = [root, root / "results", root / "figures"]
    if output in protected or any(path in output.parents for path in protected[1:]):
        parser.error("Elegir otra salida: las tablas y figuras publicadas se conservan.")
    output.mkdir(parents=True, exist_ok=True)
    print("DEMOSTRACIÓN SINTÉTICA" if args.demo else "EVALUACIÓN CON DATOS PRIVADOS", flush=True)
    # La base de trabajo y los modelos temporales se eliminan al terminar.
    with tempfile.TemporaryDirectory(prefix="ea_work_", dir=output) as temporary:
        work = Path(temporary)
        con = duckdb.connect(str(work / "work.duckdb"))
        try:
            con.execute("SET threads=2; SET memory_limit='1200MB';")
            load_inputs(con, args)
            prepare_features(con)
            candidates, selected = train_and_select(con, work)
            # Selección y umbrales quedan guardados antes de puntuar julio.
            selection = {"selected_algorithm": selected, "horizon_minutes": HORIZON,
                         "demo": args.demo, "seed": SEED, "n_features": 42,
                         "selection_rule": "Mayor average precision de junio",
                         "threshold_rule": "Mayor F1 por fila de junio", "candidates": candidates}
            (output / "seleccion.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
            evaluate_test(con, work, candidates, selected, output, outcome_end, args.demo)
        finally:
            con.close()
    print(f"Completado. Modelo elegido en validación: {LABELS[selected]}. Salida: {output}")


if __name__ == "__main__":
    main()
