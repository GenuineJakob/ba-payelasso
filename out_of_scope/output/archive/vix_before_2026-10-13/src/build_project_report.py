"""Build a local project overview and results page from existing outputs.

No models are trained and no source data or saved experiment outputs change.
Usage from Code: python src/build_project_report.py
Requires numpy, pandas and matplotlib in the reporting environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POOL_LABELS = {
    "current_monthly": "Monthly baseline (9)", "current_quarterly": "Quarterly baseline (9)",
    "fredmd_macro": "FRED-MD macro (99)", "fredmd_no_iv": "Macro + financial, without VIX (124)",
    "fredmd_full": "Full FRED-MD, including VIX (125)",
}
PREDICTOR_NAMES = {
    "cp": "Commercial-paper spread", "dfr": "Annual default-return spread", "dfy": "Default yield spread",
    "ip": "Industrial-production growth", "ipvol": "IP growth volatility", "npy": "Net payout yield",
    "ppivol": "PPI inflation volatility", "tms": "Term spread",
}


class ReportData:
    def __init__(self, root):
        self.root = root
        self.sources = {}
        self.rows = []
        self.metric_checks = 0

    def path(self, relative):
        path = self.root / relative
        self.sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def csv(self, relative, dated=False):
        frame = pd.read_csv(self.path(relative))
        if dated:
            if "date" not in frame:
                frame = frame.rename(columns={frame.columns[0]: "date"})
            frame["date"] = pd.to_datetime(frame.date)
        return frame

    def json(self, relative):
        return json.loads(self.path(relative).read_text())

    def add(self, row, predictions, column, family, specification, period, scale="level",
            window="240 months", predictors=0, alpha_grid="", source="", diagnostic=False,
            target_offset="month"):
        predictions = predictions.sort_values("date")
        y = predictions.actual.to_numpy()
        forecast = predictions[column].to_numpy()
        benchmark = predictions.benchmark.to_numpy()
        if not np.isfinite(np.column_stack([y, forecast, benchmark])).all():
            raise ValueError(f"Nonfinite prediction in {family}, {specification}.")
        mse, ar_mse = np.mean((y - forecast)**2), np.mean((y - benchmark)**2)
        rmse, improvement = np.sqrt(mse), 1 - mse / ar_mse
        np.testing.assert_allclose([mse, rmse], [row.MSE, row.RMSE], rtol=1e-9, atol=1e-12)
        metric_key = "error_reduction_vs_AR" if "error_reduction_vs_AR" in row.index else "OOS_R2"
        if pd.notna(row[metric_key]):
            np.testing.assert_allclose(improvement, row[metric_key], rtol=1e-9, atol=1e-12)
        self.metric_checks += 1
        model = "AR" if column == "benchmark" or str(row.model).startswith("ar") else "Lasso" if "lasso" in str(row.model) else "OLS"
        first, last = predictions.date.iloc[0], predictions.date.iloc[-1]
        if "target_date" in predictions:
            target_first = pd.Timestamp(predictions.target_date.iloc[0])
            target_last = pd.Timestamp(predictions.target_date.iloc[-1])
        else:
            offset = pd.offsets.MonthEnd(1) if target_offset == "month" else pd.offsets.QuarterEnd(startingMonth=12)
            target_first, target_last = first + offset, last + offset
        self.rows.append({
            "family": family, "specification": specification, "model": model, "scale": scale,
            "period": period, "window": window, "n_predictors": predictors, "n": len(predictions),
            "first_origin": first.strftime("%Y-%m-%d"), "last_origin": last.strftime("%Y-%m-%d"),
            "first_target": target_first.strftime("%Y-%m-%d"), "last_target": target_last.strftime("%Y-%m-%d"),
            "MSE": float(mse), "RMSE": float(rmse), "improvement": float(improvement),
            "alpha_grid": alpha_grid, "diagnostic": diagnostic, "source": source,
        })

    def load_results(self):
        for design in ["fixed", "rolling"]:
            table_name = f"output/tables/{design}_model_comparison.csv"
            table = self.csv(table_name)
            predictions = self.csv(f"output/results/{design}_predictions.csv", dated=True)
            for _, row in table.iterrows():
                self.add(row, predictions, row.model, f"Original {design}", "Ten macro predictors" if row.model != "benchmark" else "AR(6)",
                         "Saved evaluation", window="Fixed fit through 2000" if design == "fixed" else "360 months",
                         predictors=0 if row.model == "benchmark" else 10, source=table_name)
        table_name = "output/tables/paye_monthly_individual_model_comparison.csv"
        table = self.csv(table_name)
        predictions = self.csv("output/results/paye_monthly_individual_predictions.csv", dated=True)
        for _, row in table.iterrows():
            group = predictions.loc[predictions.paye_predictor == ("cp" if row.paye_predictor == "benchmark" else row.paye_predictor)]
            column = "benchmark" if row.paye_predictor == "benchmark" else "lasso" if "lasso" in row.model else "ols"
            self.add(row, group, column, "Paye individual", PREDICTOR_NAMES.get(row.paye_predictor, "AR(6)"),
                     "2001–2019", predictors=0 if column == "benchmark" else 1, source=table_name)
        table_name = "output/tables/paye_monthly_kitchen_sink_model_comparison.csv"
        predictions = self.csv("output/results/paye_monthly_kitchen_sink_predictions.csv", dated=True)
        for _, row in self.csv(table_name).iterrows():
            column = "benchmark" if row.model == "ar6_benchmark" else "lasso" if "lasso" in row.model else "ols"
            self.add(row, predictions, column, "Paye joint", "Eight available Paye predictors" if column != "benchmark" else "AR(6)",
                     "2001–2019", predictors=0 if column == "benchmark" else 8, source=table_name)
        table_name = "output/tables/paye_monthly_batch_experiment_results.csv"
        predictions = self.csv("output/results/paye_monthly_batch_predictions.csv", dated=True)
        for _, row in self.csv(table_name).iterrows():
            group = predictions.loc[(predictions.experiment == row.experiment) & (predictions.block == row.block)]
            group = group.loc[group.alpha_grid == (row.alpha_grid if row.model == "lasso" else "moderate_1e-4_1e0")].copy()
            benchmark = self.csv(f"output/results/paye_monthly_ar6_benchmark_predictions_{row.experiment}_window_240.csv", dated=True).set_index("date")
            group["benchmark"] = group.date.map(benchmark.benchmark)
            np.testing.assert_allclose(group.actual, group.date.map(benchmark.actual), atol=1e-12)
            column = "benchmark" if row.model == "ar6_benchmark" else row.model
            self.add(row, group, column, "Paye batch", row.block,
                     "1993–2010" if row.experiment == "paye_replication" else "2001–2019",
                     predictors=0 if column == "benchmark" else len(row.predictors.split(",")),
                     alpha_grid=str(row.alpha_grid) if pd.notna(row.alpha_grid) else "", source=table_name)
        for kind, periods in [("fredmd", ["main", "holdout"]),
                              ("contemporaneous", ["main", "later"]), ("quarterly", ["main", "later"])]:
            for scale in (["level"] if kind == "fredmd" else ["log", "level"]):
                for period in periods:
                    label = period if kind == "fredmd" else f"{scale}_{period}"
                    table_name = f"output/tables/{kind}_model_comparison_{label}.csv"
                    predictions = self.csv(f"output/results/{kind}_predictions_{label}.csv", dated=True)
                    for _, row in self.csv(table_name).iterrows():
                        group = predictions.loc[(predictions.pool == row.pool) & (predictions.model == row.model)].copy()
                        group["forecast"] = group.prediction
                        description = POOL_LABELS[row.pool]
                        if row.model.startswith("individual__"):
                            description = row.model.removeprefix("individual__core__").replace("_", " ")
                        self.add(row, group, "forecast", {"fredmd": "FRED-MD forecast", "contemporaneous": "Same-month explanation", "quarterly": "Quarterly forecast"}[kind],
                                 description, "Main" if period == "main" else "Later", scale,
                                 window="80 quarters" if kind == "quarterly" else "240 months",
                                 predictors=int(row.n_predictors), source=table_name,
                                 diagnostic=bool(row.get("diagnostic_only", False)), target_offset="quarter" if kind == "quarterly" else "month")
        self.frame = pd.DataFrame(self.rows)
        return self.frame


def pct(value):
    return f"{value * 100:+.3f}%"


def compact(frame, include_period=True):
    columns = ["specification", "model"] + (["scale", "period"] if include_period else []) + ["n", "MSE", "RMSE", "improvement"]
    output = frame[columns].copy()
    output["improvement"] = output.improvement.map(pct)
    for column in ["MSE", "RMSE"]:
        output[column] = output[column].map(lambda x: f"{x:.6f}")
    return output.rename(columns={"specification": "Specification", "model": "Model", "scale": "Scale", "period": "Period",
                                  "n": "N", "improvement": "Error reduction vs AR"})


class Document:
    def __init__(self, title):
        self.title = title
        self.blocks = []

    def heading(self, text, level=2):
        self.blocks.append(("heading", (text, level)))

    def paragraph(self, text):
        self.blocks.append(("paragraph", text))

    def bullets(self, items):
        self.blocks.append(("bullets", items))

    def code(self, text):
        self.blocks.append(("code", text))

    def table(self, frame):
        self.blocks.append(("table", frame))

    def chart(self, path, caption):
        self.blocks.append(("chart", (path, caption)))

    def markdown(self):
        parts = ["# " + self.title]
        for kind, value in self.blocks:
            if kind == "heading":
                parts.append("#" * value[1] + " " + value[0])
            elif kind == "paragraph":
                parts.append(value)
            elif kind == "bullets":
                parts.append("\n".join("- " + item for item in value))
            elif kind == "code":
                parts.append("```text\n" + value + "\n```")
            elif kind == "chart":
                parts.append(f"![{value[1]}]({value[0]})\n\n{value[1]}")
            elif kind == "table":
                def cell(v):
                    return str(v).replace("|", "\\|").replace("\n", " ")
                parts.append("| " + " | ".join(map(cell, value.columns)) + " |\n| " + " | ".join(["---"] * len(value.columns)) + " |\n" +
                             "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in value.itertuples(index=False, name=None)))
        return "\n\n".join(parts) + "\n"

    def html(self, prefix):
        parts = []
        for i, (kind, value) in enumerate(self.blocks):
            if kind == "heading":
                parts.append(f'<h{value[1]} id="{prefix}-{i}">{escape(value[0])}</h{value[1]}>')
            elif kind == "paragraph":
                parts.append("<p>" + inline(value) + "</p>")
            elif kind == "bullets":
                parts.append("<ul>" + "".join("<li>" + inline(item) + "</li>" for item in value) + "</ul>")
            elif kind == "code":
                parts.append("<pre><code>" + escape(value) + "</code></pre>")
            elif kind == "chart":
                parts.append(f'<figure><img src="{escape(value[0], quote=True)}" alt="{escape(value[1], quote=True)}"><figcaption>{escape(value[1])}</figcaption></figure>')
            elif kind == "table":
                parts.append('<div class="table-wrap">' + value.to_html(index=False, border=0, classes="data-table", escape=True) + "</div>")
        return "\n".join(parts)


def inline(text):
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", text)


def build_overview(data, generated):
    doc = Document("Project overview: stock-market volatility, macroeconomics and Lasso")
    doc.paragraph(f"Report generated {generated}. This overview describes the current code and saved experiments. The report builder reads existing outputs; it does not refit models or alter input data. The companion [results report](results_report.md) and [browsable page](results.html) contain the numerical comparisons.")
    doc.heading("1. Research question and what has been learned")
    doc.paragraph("The project asks whether macroeconomic and financial information improves predictions of S&P 500 realized volatility beyond the persistence already captured by past volatility. Its central comparison is an autoregressive benchmark against the same autoregressive block augmented with OLS or Lasso predictors. The additional experiments distinguish prediction of future volatility from explanation of volatility in the same period, and monthly from quarterly frequency.")
    doc.paragraph("The saved experiments show small and unstable incremental monthly forecast gains from additional predictors. Broader FRED-MD macro pools do not produce a consistent advantage in the later period. Same-month financial indicators, particularly when implied volatility is included, explain current volatility much better than the AR benchmark. Quarterly forecasting does not deliver a consistent improvement. These conclusions apply to the specific samples, transformations, delays and linear models tested; they do not prove that macroeconomic conditions are unrelated to volatility.")
    doc.paragraph("Paye's research motivates the incremental-information question and the lag orders: AR(6) for monthly data and AR(2) for quarterly data. His main target is log annualized volatility, with a different return construction and historical sample. The older project models retain levels; the new contemporaneous and quarterly experiments evaluate both logs and levels. The experiment named `paye_replication` is therefore a Paye-inspired comparison, not an exact replication. [Paye paper](https://vtechworks.lib.vt.edu/bitstream/handle/10919/78015/Dejavol_prepub.pdf?sequence=1).")

    doc.heading("2. Inputs and data coverage")
    sp = data.csv("data/raw/sp500.csv")
    macro = data.csv("data/raw/fred_macro.csv")
    base = data.csv("data/processed/dataset.csv", dated=True)
    snapshot = data.csv("data/raw/fredmd_snapshot_2026-09.csv")
    doc.table(pd.DataFrame([
        ["S&P 500 daily prices", f"{len(sp):,} daily rows; {sp.iloc[0,0]} to {sp.iloc[-1,0]}", "Daily log index returns and monthly/quarterly realized volatility"],
        ["Original FRED macro cache", f"{len(macro)} monthly dates; 1970–2023; 12 source series", "Rates, spreads, prices, production, unemployment, quarterly GDP"],
        ["Processed original monthly dataset", f"{len(base)} consecutive month ends; {base.date.min():%Y-%m} to {base.date.max():%Y-%m}", "Volatility, next-month target, 14 candidate explanatory columns"],
        ["Goyal–Welch annual data", "Optional cached annual source", "Default-return spread, consumption/wealth ratio, investment/capital ratio"],
        ["Cached payout dataset", "Monthly payout observations; processed coverage ends in 2019", "Net payout yield in the older Paye models"],
        ["FRED-MD September 2026 vintage", f"{len(snapshot.columns)-1} series; {len(snapshot)-1} observations starting January 1959", "Broad monthly macroeconomic and financial information"],
    ], columns=["Input", "Coverage", "Role"]))
    doc.paragraph("`download_data.py` currently uses tidyfinance to obtain the S&P index and FRED observations, then writes the established CSV formats. External annual and payout sources are read from local caches by the dataset builder. `download_fredmd.py` downloads a pinned September 2026 snapshot and official variable appendix, requires no API key for those files, and checks cached hashes. It does not silently advance to a newer vintage. [Official FRED-MD download page](https://www.stlouisfed.org/research/economists/mccracken/fred-databases).")
    doc.paragraph("The FRED-MD snapshot extends beyond the modeling sample. Its later observations are not used to extend the S&P target beyond 2023. Source files, processed datasets and build manifests remain separate. The reporting manifest lists hashes of every file used to produce this report.")

    doc.heading("3. Volatility and target construction")
    doc.code("Daily return: r_d = log(P_d) − log(P_(d−1))\nMonthly annualized volatility: Vol_m = sqrt(12 × Σ_(d in m) r_d²)\nQuarterly annualized volatility: Vol_q = sqrt(4 × Σ_(d in q) r_d²)\nLog specification: y_t = log(Vol_t); level specification: y_t = Vol_t")
    doc.paragraph("The code uses the cached `Close` price field, falling back to `Adj Close` where supported. The current downloader maps its adjusted close to `Close`. This is the existing S&P index-return construction, not Paye's daily excess-return construction. A volatility value of 0.20 represents annualized volatility of 20%. Realized volatility here is the square root of a sum of squared daily returns, with annualization; it is not the sample standard deviation of monthly returns.")
    doc.paragraph("Returns are computed before aggregation, so a return crossing a month or quarter boundary is assigned to its ending date. Quarterly volatility is built directly from daily returns, not by averaging monthly volatilities. For complete matching quarters, `Vol_q² = (Vol_m1² + Vol_m2² + Vol_m3²)/3`. The quarterly builder excludes the first quarter because its opening return is unavailable; absent calendar months and clearly early terminal quarters are marked unavailable. It does not reconstruct the exchange's entire trading calendar.")
    doc.paragraph("The original and FRED-MD forecast datasets index rows by origin month t and put `Vol_(t+1)` in `target`. The contemporaneous dataset instead puts `Vol_t` in `target`, with controls `Vol_(t−1)` through `Vol_(t−6)`. The quarterly dataset indexes the origin q and predicts `Vol_(q+1)` from `Vol_q` and `Vol_(q−1)`. New prediction outputs explicitly store `target_date`. Log runs transform the response and every AR term consistently and reject nonpositive observed volatility instead of adding an offset.")

    doc.heading("4. Predictors and timing")
    doc.table(pd.DataFrame([
        ["Unemployment", "UNRATE level", "Two months", "Previous quarter"],
        ["Federal funds rate", "FEDFUNDS level", "Current month", "Current quarter"],
        ["Default yield spread", "BAA − AAA", "Current month", "Current quarter"],
        ["Term spread", "GS10 − TB3MS", "Current month", "Current quarter"],
        ["Commercial-paper spread", "Modern CP series, historical fallback, minus TB3MS", "Current month", "Current quarter"],
        ["Inflation", "12-month CPI log change; quarterly: four-quarter log change of quarterly mean CPI", "Two months", "Previous quarter"],
        ["IP growth", "12-month IP log change; quarterly: four-quarter log change of quarterly mean IP", "Two months", "Previous quarter"],
        ["IP volatility", "12-month SD of annual IP growth; quarterly: four-quarter SD of annual growth", "Two months", "Previous quarter"],
        ["PPI volatility", "12-month SD of annual PPI inflation; quarterly: four-quarter SD of annual inflation", "Two months", "Previous quarter"],
    ], columns=["Nine-variable baseline", "Construction", "FRED-MD monthly forecast delay", "Quarterly forecast availability"]))
    doc.paragraph("Quarterly index, rate and spread means require three finite monthly observations. An incomplete block becomes missing and is later imputed using the training window only. Growth transformations are applied to quarterly mean indices; volatility measures use the latest four annual-growth observations and the sample standard deviation with `ddof=1`. Modern commercial-paper data take precedence, with historical values used where modern observations are missing.")
    doc.paragraph("The older Paye menu has eight available predictors: cp, dfr, dfy, ip, ipvol, npy, ppivol and tms. `exret` is absent and is not replaced with zero. The processed dataset also contains GDP, cay and ik, but these do not all enter the Paye menu. GDP is quarterly carried into monthly rows; dfr, cay and ik come from annual data carried forward. These are not new monthly observations. The new baseline excludes those repeated low-frequency inputs and excludes npy because its coverage ends in 2019. The original ten-predictor models retain GDP as part of their historical specification.")
    doc.table(pd.DataFrame([
        ["Monthly baseline", 9, "The nine variables listed above"],
        ["FRED-MD macro", 99, "Official groups 1–5 and 7: activity, labor, housing, consumption/orders, money/credit and prices"],
        ["FRED-MD without implied volatility", 124, "Macro plus rates, exchange rates and stock indicators; excludes VIXCLSx"],
        ["Full FRED-MD", 125, "Identical broad financial pool plus VIXCLSx"],
    ], columns=["Pool", "Eligible extra predictors", "Content"]))
    doc.paragraph("FRED-MD transformations come from the pinned CSV's transformation row: level, first difference, second difference, log level, first log difference, second log difference, or change in fractional growth. They use only the current and earlier observations. Invalid logs and divisions become missing, with no future backfill or interpolation. [Transformation documentation](https://files.stlouisfed.org/files/htdocs/wp/2015/2015-012.pdf).")
    for name in ["fredmd", "contemporaneous", "quarterly"]:
        data.path(f"data/processed/{name}_feature_metadata.csv")
    doc.paragraph("The complete variable names, descriptions, transformations, eligibility decisions and delays are available in the [monthly forecast feature dictionary](../../data/processed/fredmd_feature_metadata.csv), [same-month feature dictionary](../../data/processed/contemporaneous_feature_metadata.csv), and [quarterly feature dictionary](../../data/processed/quarterly_feature_metadata.csv).")
    doc.paragraph("For forecasting, broad FRED-MD macro indicators have a two-month assumed delay; rates, exchange rates, S&P prices and implied volatility use the origin month, while S&P dividend yield and P/E have a six-month delay. The same-month explanation removes these shifts and uses reference-month observations ex post. Original/Paye forecasts do not implement this explicit publication-delay schedule, so those saved outputs should be interpreted as revised-data comparisons rather than historical real-time forecasts.")

    doc.heading("5. Models and why the AR block is unpenalized")
    doc.code("Monthly forecast: y_(t+1) = a + Σ_(k=0 to 5) φ_(k+1)y_(t−k) + β′X_t + ε_(t+1)\nSame-month explanation: y_t = a + Σ_(k=1 to 6) φ_k y_(t−k) + β′X_t + ε_t\nQuarterly forecast: y_(q+1) = a + φ_1 y_q + φ_2 y_(q−1) + β′X_q^available + ε_(q+1)")
    doc.bullets([
        "AR benchmark: estimate the intercept and volatility lags by ordinary least squares, with no additional predictors.",
        "AR + OLS: estimate the AR block and additional predictors jointly without regularization. Adding variables can improve training fit while worsening evaluation performance.",
        "AR + Lasso: preserve the intercept and complete AR block; penalize only additional predictors. A sufficiently large penalty produces the corresponding AR benchmark.",
    ])
    doc.paragraph("The shared `PartiallyPenalizedLassoCV` estimator projects both the response and extra predictors on the AR controls, standardizes the remaining predictor residuals, estimates the Lasso coefficients, then recovers the intercept and AR coefficients by conditional OLS. Coefficients are returned in the original predictor units and in the declared response scale. A zero extra coefficient means no incremental contribution in that fitted model, not that the variable has no economic relationship with volatility.")
    doc.paragraph("Alpha is selected using five chronological validation folds and complete AR-plus-Lasso validation predictions. Older/general experiments use 200 alphas from 10⁻⁵ to 10; the batch compares 200-value grids from 10⁻⁴ to 1 and from 10⁻³ to 1. Older runs use coordinate descent. FRED-MD and the new analyses use a checked LARS path evaluated on the same requested grid, with numerical polishing if necessary. Internal target normalization is reversed consistently, preserving the penalized objective. Solver diagnostics record relative dual gaps and convergence warnings.")
    doc.paragraph("In the new experiments, remaining predictor gaps are filled with training-window medians. Inside every validation fold, imputation, AR residualization and scaling are fitted only on that fold's training observations. All-missing or constant extra predictors have zero coefficients. AR controls and responses must be finite and are never imputed. Feature eligibility is decided once from at least 95% finite observations in the initial 240-month or 80-quarter window, not from later coverage or correlations with evaluation targets. `ACOGNO`, which starts in 1992, fails that initial coverage check.")
    doc.paragraph("The older original and Paye workflows instead use common complete-case samples. Requiring every available Paye variable makes even one-variable comparisons end in December 2019 because of npy. This keeps those models comparable within that experiment but sacrifices later observations. Dropping dates could also turn an observation-count rolling window into something other than consecutive calendar months; the new workflows explicitly preserve calendar rows.")

    doc.heading("6. Experiment designs and actual evaluation dates")
    designs = data.frame[["family", "period", "scale", "window", "n", "first_origin", "last_origin", "first_target", "last_target"]].drop_duplicates()
    doc.table(designs.rename(columns={"family": "Family", "period": "Period", "scale": "Scale", "window": "Training design", "n": "N per model", "first_origin": "First origin", "last_origin": "Last origin", "first_target": "First response", "last_target": "Last response"}))
    doc.paragraph("The original fixed model is trained once through December 2000 and supplies 240 evaluation origins. Its rolling counterpart needs 360 complete monthly rows, so its saved evaluation begins in December 2001 and contains 229 origins. Paye individual and joint runs use 240-month windows with 228 origins in 2001–2019. The batch uses seven predictor blocks and two alpha grids in each of two historical periods; its best row is a retrospective selection across tested configurations.")
    doc.paragraph("FRED-MD and contemporaneous analyses use 240 main and 35 later monthly observations. Quarterly analyses use 80 main and 12 later response quarters. Forecast origin dates and response dates differ: the FRED-MD main origins January 2001–December 2020 predict February 2001–January 2021; the same-month analysis explains January 2001–December 2020 itself. Quarterly main origins 2000Q4–2020Q3 predict 2001Q1–2020Q4. These differences matter when interpreting crises and comparing periods.")
    doc.paragraph("The filenames call the FRED-MD later period `holdout`; subsequent analysis has already inspected that period. It is now reported here as later-period robustness, not a fresh untouched test. Rolling re-estimation remains chronological, but later model choices are informed by observed results. Small quarterly later-period improvements rest on only 12 observations.")

    doc.heading("7. Evaluation, reproducibility and limits")
    doc.code("MSE = mean[(actual − prediction)²]\nRMSE = sqrt(MSE)\nError reduction vs AR = 1 − MSE_model / MSE_AR\nDisplayed percent = 100 × error reduction vs AR")
    doc.paragraph("Positive error reduction means better accuracy than the matching AR model. For example, 0.0042 in a CSV means about 0.42% lower MSE, not 42%. This relative measure is not conventional R-squared against a historical mean and is not the fraction of total volatility explained. In levels, an RMSE of 0.0897 means about 8.97 annualized volatility percentage points. Log and level MSEs have different units and must not be ranked directly. No exponentiation or mean-bias adjustment is applied to log predictions.")
    doc.paragraph("The saved model comparisons describe realized sample performance. The current project does not report forecast-comparison significance tests, confidence intervals or causal identification. OLS now centers and standardizes predictors using only each training window and solves with an explicit relative singular-value cutoff, `eps(float64) * max(n_train, n_predictors)`. Coefficients are restored to original units. A numerical-rank report depends on this cutoff and scaling; it does not itself prove mathematical dependence. The corrected broad fits have full numerical rank, while their forecast errors remain large.")
    doc.paragraph("The earlier implementation used scikit-learn's dense OLS default, which in version 1.9 uses a `1e-6` relative singular-value cutoff. On unscaled data this discarded independent small directions. In the first full FRED-MD window, raw/conventionally scaled numerical rank is 131, while the old cutoff reported 125. The corrected standardized condition number is about 1,081, compared with about 10.3 million for the centered raw design. This is a numerical estimation issue, not evidence that broad OLS is inherently unidentified. [Scikit-learn tolerance documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).")
    doc.paragraph("The fixed FRED-MD vintage and original cached macro data are revised data. Assumed reporting delays do not undo later revisions or reconstruct the information set at every historical date. The volatility source differs from Paye's excess-return construction; the stock index target is not implied volatility. A high contemporaneous VIX selection rate is consistent with a strong observed market relationship but is not evidence that VIX improves next-month forecasts.")
    validation = data.json("output/tables/volatility_analysis_validation.json")
    doc.paragraph(f"The volatility-analysis validation records {validation['tests_passed']} passing tests and independent reconstruction of {validation['total_reconstructed_predictions']:,} same-month/quarterly model predictions, including training R-squared and metrics. The OLS correction additionally checked 17,058 prediction rows across FRED-MD and those analyses, retaining AR/Lasso unchanged while reestimating 10,090 full-evaluation OLS fits. The report recalculates metrics for all {data.metric_checks} canonical comparison rows. These numerical checks are separate from testing statistical significance.")

    doc.heading("8. Files, entry points and how to reproduce")
    doc.table(pd.DataFrame([
        ["Raw input acquisition", "download_data.py; download_fredmd.py", "Existing price/macro CSVs; pinned FRED-MD CSV, appendix and checksum manifest"],
        ["Dataset construction", "build_dataset.py; build_fredmd_dataset.py; build_volatility_analysis_datasets.py", "Original, delayed FRED-MD, same-month and quarterly datasets"],
        ["Original models", "train_models.py", "Fixed and 360-month rolling benchmark/OLS/Lasso"],
        ["Paye models", "train_paye_monthly_individual.py; train_paye_monthly_kitchen_sink.py; train_paye_monthly_batch_experiments.py", "Individual, joint and historical block/grid comparisons"],
        ["Expanded models", "train_fredmd.py", "Four delayed monthly predictor pools"],
        ["New questions", "train_contemporaneous.py; train_quarterly.py", "Same-month explanation and next-quarter forecasts in both scales"],
        ["Shared estimation", "partially_penalized_lasso.py; paye_monthly_utils.py; volatility_analysis_utils.py", "Unpenalized controls, tuning, imputation, exports and metrics"],
        ["Reporting", "evaluate_paye_monthly_outputs.py; build_project_report.py", "Legacy output summaries and this comprehensive local report"],
    ], columns=["Stage", "Scripts under src/", "Purpose"]))
    doc.paragraph("Run commands from the `Code` directory in an environment with the project dependencies. The FRED-MD requirements cover its models and tests; the reporting environment additionally needs matplotlib. Original downloading additionally needs tidyfinance. Re-running training replaces that experiment's own output files; re-running only this report reads them without training. Preserve the pinned caches if you want the same source snapshot.")
    doc.code("# Original comparisons\npython src/train_models.py\n# Paye-inspired individual, joint and block experiments\npython src/train_paye_monthly_individual.py\npython src/train_paye_monthly_kitchen_sink.py\npython src/train_paye_monthly_batch_experiments.py\n# Expanded monthly forecasting\npython src/build_fredmd_dataset.py\npython src/train_fredmd.py --period all\n# Same-month explanation and quarterly forecasting\npython src/build_volatility_analysis_datasets.py\npython src/train_contemporaneous.py --period all\npython src/train_quarterly.py --period all\n# Refresh this report without rerunning models\npython src/build_project_report.py")
    doc.paragraph("For the new runners, `--max-forecasts 2` produces separate smoke outputs; these are excluded from this report. Logs and levels are selected with `--scale log|level|both`; new experiment periods use `main|later|all`, while the older FRED-MD runner retains `main|holdout|all`. Dataset-building source checksums guard against silently changing the frozen inputs.")
    doc.paragraph("For definitions and detailed runner options, see [FRED-MD instructions](../../README_fredmd.md) and [volatility-analysis instructions](../../README_volatility_analysis.md). Existing results live in `output/results/` and `output/tables/`; new reporting artifacts live in `output/reports/`.")
    doc.heading("9. Possible next research steps, not yet implemented")
    doc.bullets([
        "Keep forecasting and contemporaneous explanation as separate research questions. Interpret the financial same-month results as an association, with the timing stated explicitly.",
        "A monthly log-volatility forecasting experiment would make the forecasting target transformation closer to the new primary specifications. The broad monthly FRED-MD forecasts currently use levels only.",
        "Predefine a small number of forecast combinations or factor-based macro models before evaluating new data. Do not select a specification simply because it happens to win among the inspected configurations.",
        "If making inferential claims, add an appropriate time-series forecast-comparison design and uncertainty estimates; the current error tables alone do not establish significance.",
        "For genuine real-time forecasting, obtain historical vintages and publication dates. Conservative delays on a revised snapshot are an approximation, not a substitute for those data.",
    ])
    return doc


def source_links(frame):
    return "Sources: " + "; ".join(f"[{Path(p).name}](../../{p})" for p in frame.source.drop_duplicates()) + "."


def build_results(data, generated):
    frame = data.frame
    doc = Document("Detailed results: volatility forecasting and explanation")
    doc.paragraph(f"Generated {generated} from saved full evaluations. All {len(frame)} canonical comparison rows were independently reconciled to prediction errors. Smoke runs and derived best-row inventories are excluded. [Project overview](project_overview.md) explains the construction and assumptions; [interactive page](results.html) provides filters, charts and source links.")
    doc.heading("1. How to read the results")
    doc.paragraph("Every percentage is the reduction in MSE against that experiment's matching AR benchmark: positive is better, negative is worse. For forecasts this is out-of-sample R-squared relative to AR; for same-month explanation it is held-out explanatory performance relative to AR. It is not conventional R-squared against the sample mean. Baselines are displayed as zero reduction even where older CSVs leave their own R-squared blank.")
    doc.paragraph("Evaluate comparisons within the same experiment, dates and response scale. Raw log MSE cannot be compared with level MSE. Original rolling, Paye, FRED-MD and quarterly samples differ in dates, window sizes, timing and eligible predictors. The later period is retrospective robustness. Reported differences are sample estimates; this project has not supplied significance tests or confidence intervals.")
    headline = []
    for family, scale, specs in [
        ("FRED-MD forecast", "level", list(POOL_LABELS.values())[:1] + [POOL_LABELS[k] for k in ["fredmd_macro", "fredmd_no_iv", "fredmd_full"]]),
        ("Same-month explanation", "log", [POOL_LABELS[k] for k in ["current_monthly", "fredmd_macro", "fredmd_no_iv", "fredmd_full"]]),
        ("Quarterly forecast", "log", [POOL_LABELS["current_quarterly"]]),
    ]:
        for spec in specs:
            selected = frame.loc[(frame.family == family) & (frame.scale == scale) & (frame.specification == spec) & (frame.model == "Lasso")]
            headline.append([family, scale, spec, pct(selected.loc[selected.period == "Main", "improvement"].iloc[0]),
                             pct(selected.loc[selected.period == "Later", "improvement"].iloc[0])])
    doc.table(pd.DataFrame(headline, columns=["Question", "Scale", "Lasso predictor pool", "Main error reduction", "Later error reduction"]))
    doc.paragraph("The broad monthly forecast gains are modest and do not persist consistently later. The macro-only pool gives about 1.17% lower main-period MSE but about 0.016% lower later MSE, effectively a tie in practical terms without further uncertainty analysis. Same-month financial pools show much larger gains in both periods. Quarterly Lasso's small main log advantage does not persist later or in levels.")

    doc.heading("2. Original fixed and 360-month rolling models")
    original = frame.loc[frame.family.str.startswith("Original")]
    doc.paragraph("The fixed model has 240 forecast origins, January 2001–December 2020. It fits once through December 2000. The 360-month rolling model has 229 saved origins, December 2001–December 2020, because the complete-case history limits the first possible window. Both use six AR terms and ten macro predictors, including the carried-forward quarterly GDP variable. Targets are next-month annualized volatility in levels.")
    doc.table(compact(original.assign(specification=original.family)))
    doc.paragraph("Fixed Lasso collapses to the AR benchmark; joint OLS raises MSE by about 7.98%. Rolling Lasso is almost identical to AR, with a 0.042% increase in MSE, while OLS raises MSE by about 4.19%. These two designs do not indicate meaningful added forecast value from this ten-variable set.")
    doc.paragraph(source_links(original))

    doc.heading("3. Paye individual predictors and the joint model")
    individual = frame.loc[frame.family == "Paye individual"]
    joint = frame.loc[frame.family == "Paye joint"]
    doc.paragraph("All these fits use a common 240-month rolling sample and 228 origins in January 2001–December 2019. Net payout yield's coverage causes that endpoint even for other individual predictors. Eight extra predictors are available; `exret` is missing. Dfr is sourced from annual observations, so it should not be interpreted as a new monthly bond-return signal.")
    doc.table(compact(individual))
    doc.table(compact(joint))
    doc.paragraph("The largest individual Lasso gains are IP volatility at about 0.057% and the annual default-return spread at about 0.056%. Every individual OLS fit is worse than AR. Joint Lasso improves MSE by about 0.021%; joint OLS worsens it by 16.43%. Lasso mainly protects the AR specification from the instability of adding predictors, rather than delivering a substantial accuracy gain.")
    doc.paragraph(source_links(pd.concat([individual, joint])))

    doc.heading("4. Paye batch: blocks and penalty grids")
    batch = frame.loc[frame.family == "Paye batch"]
    best = batch.loc[batch.model == "Lasso"].sort_values("improvement", ascending=False).groupby("period", sort=False).head(1)
    doc.table(compact(best).assign(**{"Alpha grid": best.alpha_grid.to_numpy()}))
    doc.paragraph("The best saved Lasso configuration for 1993–2010 is the all-available block with the moderate grid, reducing MSE by about 0.420%. The best 2001–2019 configuration is the credit block with the stronger grid, reducing MSE by about 0.13%. Both are small. Their ranking is retrospective across seven blocks and two alpha grids, not performance of a single prespecified winner on a fresh test.")
    doc.paragraph("The blocks are best_individuals (dfy, ipvol, npy), best_individuals_plus_ip, credit (cp, dfr, dfy), real_activity (ip, ipvol, ppivol), return_payout (npy only because exret is absent), credit_plus_activity, and all_available. All use 240-month windows. Repeated AR rows below are the same benchmark carried into each block comparison, not distinct models. The moderate grid spans 10⁻⁴–1; the stronger grid spans 10⁻³–1, each with 200 values.")
    doc.table(compact(batch).assign(**{"Alpha grid": batch.alpha_grid.to_numpy()}))
    doc.paragraph(source_links(batch))

    doc.heading("5. Broad FRED-MD monthly forecasting")
    fred = frame.loc[frame.family == "FRED-MD forecast"]
    doc.chart("assets/fredmd_forecast.png", "Lasso error reduction relative to AR(6), levels: main and later monthly forecast periods.")
    doc.paragraph("This tests whether considerably more predictors improve next-month forecasts. The eligible pools have 9, 99, 124 and 125 extra predictors, all with the same AR(6) benchmark, 240-month window, training-only preprocessing and identical dates. The target remains in levels. Macroeconomic inputs have an assumed two-month delay; rates, exchange rates, S&P prices and VIX are contemporaneously available at the origin; dividend yield and P/E are delayed six months.")
    doc.table(compact(fred))
    doc.paragraph("The macro-only Lasso pool performs best in the main period, at 1.171% lower MSE. Its later gain is only 0.016%. Adding financial variables gives 0.161% main improvement without VIX and 0.188% with VIX, followed by a 0.494% deterioration later for both. The nine-variable baseline deteriorates by 9.899% later. These results do not support a substantial, stable forecasting benefit from expanding the monthly macro pool.")
    corrected_ols = fred.loc[(fred.model == "OLS") & (fred.period == "Main")]
    macro_ols = corrected_ols.loc[corrected_ols.specification == POOL_LABELS["fredmd_macro"]].iloc[0]
    full_ols = corrected_ols.loc[corrected_ols.specification == POOL_LABELS["fredmd_full"]].iloc[0]
    doc.paragraph(f"After correcting training-window scaling and the numerical cutoff, main macro-only OLS MSE is {macro_ols.MSE:.6f} and full-pool OLS MSE is {full_ols.MSE:.6f}. Full-pool OLS has {-full_ols.improvement*100:.2f}% higher MSE than AR. All corrected joint fits have full numerical rank at the declared cutoff; poor forecast accuracy remains, but the earlier explanation involving inherently unidentified coefficients is withdrawn. Older results are archived for comparison.")
    doc.paragraph(source_links(fred))
    if (data.root / "output/tables/ols_correction_validation.json").exists():
        correction = data.json("output/tables/ols_correction_manifest.json")
        verification = data.json("output/tables/ols_correction_validation.json")
        prior_path = correction["archive"] + "/output/tables/fredmd_model_comparison_main.csv"
        earlier = data.csv(prior_path)
        comparison = []
        for pool, fitted in [("fredmd_macro", macro_ols), ("fredmd_full", full_ols)]:
            old = earlier.loc[(earlier.pool == pool) & (earlier.model == "ols")].iloc[0]
            comparison.append([POOL_LABELS[pool], f"{old.MSE:.6f}", f"{fitted.MSE:.6f}"])
        doc.heading("OLS numerical correction and provenance", level=3)
        doc.table(pd.DataFrame(comparison, columns=["Main-period pool", "Earlier unscaled OLS MSE", "Corrected standardized OLS MSE"]))
        audit = data.csv("output/tables/ols_first_window_audit.csv")
        doc.table(audit[["pool", "predictors_including_AR", "legacy_1e6_rank", "conventional_raw_rank", "corrected_scaled_rank"]].rename(columns={
            "pool":"First-window pool", "predictors_including_AR":"Predictors incl. AR", "legacy_1e6_rank":"Legacy raw rank at 1e-6",
            "conventional_raw_rank":"Conventional raw rank", "corrected_scaled_rank":"Corrected scaled rank"}))
        doc.table(pd.DataFrame(list(verification["software_versions"].items()), columns=["Corrected OLS environment", "Version / platform"]))
        doc.paragraph("These versions apply to the OLS correction. AR and Lasso predictions and coefficients were inherited unchanged; their original package versions were not recorded and are not retroactively assigned. The preserved environment's scikit-learn 1.9.1 default reproduces the archived first full-pool coefficients to about 1.8×10⁻¹⁵. Standardization uses training data only; the explicit cutoff is machine epsilon times the larger design dimension (5.329×10⁻¹⁴ in the first 240-month window).")
        doc.paragraph(f"Earlier results and reports are [archived]({ '../../' + correction['archive'] + '/archive_manifest.json' }); the [correction manifest](../tables/ols_correction_manifest.json) and [independent checks](../tables/ols_correction_validation.json) record the update. The conclusion that broad OLS forecasts poorly survives; the claim that these broad designs are inherently rank deficient does not.")

    doc.heading("6. Same-month explanation: primary log results and levels robustness")
    contemporaneous = frame.loc[frame.family == "Same-month explanation"]
    doc.chart("assets/contemporaneous_log.png", "Same-month Lasso explanatory error reduction relative to AR(6), log volatility. Current-period indicators are supplied ex post.")
    doc.paragraph("The response is current-month volatility. Its controls are the previous six months, never the current volatility itself. Models are fitted using preceding months; current-month indicators are then supplied to evaluate the fitted relationship. This held-out design tests generalization of a same-period association without turning that information into a real-time forecast.")
    doc.paragraph("In logs, financial pools reduce main-period error by 33.562% without VIX and 52.826% with VIX; later reductions are 30.550% and 49.845%. In levels, corresponding gains are 42.529% and 67.235% main, and 15.359% and 56.724% later. The financial contribution appears in both periods and both scales. The macro-only pool worsens main error by 40.218% in logs and 79.171% in levels, and does not improve later error.")
    doc.paragraph("The nine-variable baseline has small Lasso gains in the main period (0.684% log, 3.805% levels) but worsens later performance. Baseline joint OLS also reverses sign between periods. Individual indicators offer mixed associations: commercial-paper/default spreads improve main-period log explanation, whereas inflation helps later. The table includes all individual OLS comparisons and both scales; these are not independent causal-effect tests.")
    doc.table(compact(contemporaneous))
    doc.paragraph("VIXCLSx is selected in every full-pool log fit. The S&P price indicator is selected in 63.33% of main full-pool log fits and 100% later; without VIX it is selected in 91.67% main and 100% later. Selection describes conditional fitted coefficients, not statistical significance or proof of causality. Same-month S&P returns and volatility naturally summarize the same market period, so much better explanation can coexist with weak next-month forecasts.")
    doc.paragraph(source_links(contemporaneous))

    doc.heading("7. Quarterly forecasting: AR(2), individual variables, OLS and Lasso")
    quarterly = frame.loc[frame.family == "Quarterly forecast"]
    doc.chart("assets/quarterly_log.png", "Quarterly log-volatility forecast error reduction versus AR(2), with 80 main and 12 later response quarters.")
    doc.paragraph("Quarterly forecasts use the same nine-variable baseline, 80-quarter windows and two unpenalized AR terms. Macroeconomic features use the previous complete quarter; financial indicators use the origin quarter. Targets are complete quarterly volatility computed from daily returns. The main response period is 2001Q1–2020Q4 and the later response period is 2021Q1–2023Q4.")
    doc.table(compact(quarterly))
    doc.paragraph("Primary log Lasso reduces main MSE by 0.481% and raises later MSE by 0.428%. In levels, it raises MSE by 2.591% main and 5.849% later. Joint OLS raises main MSE by about 27.85% in logs and 27.70% in levels, then reduces later MSE by 15.00% and 9.34%. No individual predictor beats AR(2) in the main period on either scale. Later benefits for some variables and joint OLS rest on only 12 quarters and do not establish consistent improvement.")
    doc.paragraph(source_links(quarterly))

    doc.heading("8. What Lasso actually selects")
    doc.paragraph("Selection frequencies count only additional predictor coefficients, not intercepts or AR terms. Highly correlated variables can substitute for each other, so these are descriptive rankings. A predictor can be frequently selected yet fail to improve aggregate error. The contrast between rare selection for future forecasting and frequent selection for current-period financial explanation is particularly informative.")
    selected_rows = []
    for kind, labels in [("fredmd", ["main", "holdout"]), ("contemporaneous", ["log_main", "log_later"]), ("quarterly", ["log_main", "log_later"])]:
        for label in labels:
            frequency = data.csv(f"output/tables/{kind}_selection_frequency_{label}.csv")
            for pool, group in frequency.groupby("pool", sort=False):
                for _, row in group.sort_values("selected_share", ascending=False).head(3).iterrows():
                    selected_rows.append([kind, label, POOL_LABELS[pool], row.feature, row.description, f"{row.selected_share*100:.2f}%", int(row.selected_n)])
    doc.table(pd.DataFrame(selected_rows, columns=["Experiment", "Period/scale", "Pool", "Feature", "Description", "Windows selected", "Count"]))

    doc.heading("9. Data quality and numerical checks")
    diagnostic_rows = []
    for kind, labels in [("fredmd", ["main", "holdout"]),
                         ("contemporaneous", ["log_main", "log_later", "level_main", "level_later"]),
                         ("quarterly", ["log_main", "log_later", "level_main", "level_later"])]:
        for label in labels:
            diagnostics = data.csv(f"output/tables/{kind}_window_diagnostics_{label}.csv")
            if "model" in diagnostics:
                ols = diagnostics.loc[diagnostics.model == "joint_ols"]
                lasso = diagnostics.loc[diagnostics.model == "lasso"]
            else:
                ols, lasso = diagnostics, diagnostics
            rank_deficient = ols.ols_rank_deficient.map(lambda v: str(v).casefold() == "true").sum()
            legacy_losses = int((ols.ols_raw_rank_at_1e6 < ols.ols_fitted_columns).sum())
            diagnostic_rows.append([kind, label, len(ols), int(rank_deficient), legacy_losses, int(lasso.lasso_convergence_warnings.sum()),
                                    f"{lasso.lasso_max_cv_relative_dual_gap.max():.3g}",
                                    f"{lasso.training_imputed_fraction.mean()*100:.4f}%"])
    doc.table(pd.DataFrame(diagnostic_rows, columns=["Experiment", "Period/scale", "Joint OLS fits", "Scaled numerical-rank shortfalls", "Raw rank losses at 1e-6", "Lasso convergence warnings", "Max CV relative dual gap", "Mean training predictor imputation"]))
    doc.paragraph("Corrected standardized joint OLS fits have no numerical-rank shortfalls under the explicit conventional cutoff. The raw `1e-6` column reproduces the scale-sensitive truncation in the older solver; it must not be interpreted as exact dependence. Diagnostics include the standardized singular values, cutoff and condition number. Missing-value medians and imputation rates are exported separately. There is no clipping of negative level predictions, and poor OLS forecasts remain in the error calculations.")
    validation = data.json("output/tables/volatility_analysis_validation.json")
    doc.paragraph(f"The saved analysis validation records {validation['tests_passed']} passing tests and {validation['total_reconstructed_predictions']:,} independently reconstructed same-month/quarterly predictions. The later OLS correction verifies 17,058 prediction rows across all three affected experiment families. AR/Lasso predictions and coefficients remain unchanged versus archived outputs. All 238 current canonical comparison rows reconcile against their prediction errors. The correction reestimated OLS; ordinary report generation reads those outputs without refitting models.")
    doc.paragraph("A dual-gap tolerance is a numerical optimization check, not a confidence interval. Strong-penalty tests recover AR, weak-penalty tests approach joint OLS, and fold tests verify that future/validation observations cannot change training preprocessing. Data tests cover transformation codes, lag alignment, quarterly aggregation, missingness and calendar preservation.")

    doc.heading("10. Interpretation for the project")
    doc.bullets([
        "A useful persistence benchmark is hard to beat. The available macro predictors add little consistent incremental forecast accuracy in these saved specifications.",
        "More variables are not automatically better. Corrected wide OLS is full rank under the stated numerical cutoff but still performs poorly. Lasso limits additional coefficients without guaranteeing positive forecasting results.",
        "The same-period question yields a different result: financial information, including VIX and S&P price changes, explains current volatility substantially better than the AR benchmark.",
        "Quarterly aggregation changes the target and information horizon but has not produced a consistent forecasting advantage here.",
        "These are revised-data, linear-model, sample-specific findings. The inspected later period and retrospectively selected batch winners should not be presented as an untouched validation exercise. No forecast significance or causal claim follows from the error tables alone.",
    ])
    doc.paragraph("All underlying comparisons remain accessible through the local CSV links and the page's full-results explorer. Rebuild tables, charts and provenance with `python src/build_project_report.py`; this refreshes only reporting artifacts. Interpretation prose describes the current experiment configuration and should be reviewed whenever definitions or results materially change.")
    return doc


def build_charts(frame, output):
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ba-project-report-mpl-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for family, scale, filename, title in [
        ("FRED-MD forecast", "level", "fredmd_forecast", "Next-month forecasting • Lasso in levels"),
        ("Same-month explanation", "log", "contemporaneous_log", "Same-month explanation • Lasso in logs"),
        ("Quarterly forecast", "log", "quarterly_log", "Next-quarter forecasting • log volatility"),
    ]:
        selected = frame.loc[(frame.family == family) & (frame.scale == scale)]
        if family == "Quarterly forecast":
            selected = selected.loc[selected.model.isin(["OLS", "Lasso"]) & (selected.specification == POOL_LABELS["current_quarterly"])].copy()
            selected["chart_label"] = selected.model
        else:
            selected = selected.loc[(selected.model == "Lasso") & selected.specification.isin(POOL_LABELS.values())].copy()
            selected["chart_label"] = selected.specification
        labels = selected.chart_label.drop_duplicates().tolist()
        fig, ax = plt.subplots(figsize=(10.2, 4.5), layout="constrained")
        y = np.arange(len(labels))
        for j, (period, color) in enumerate([("Main", "#175d81"), ("Later", "#bc652b")]):
            values = [selected.loc[(selected.chart_label == label) & (selected.period == period), "improvement"].iloc[0] * 100 for label in labels]
            bars = ax.barh(y + (j - .5) * .32, values, height=.30, color=color, label=period)
            ax.bar_label(bars, labels=[f"{v:+.2f}%" for v in values], padding=5, fontsize=10)
        ax.set_yticks(y, labels, fontsize=10)
        ax.invert_yaxis()
        ax.axvline(0, color="#444", linewidth=.8)
        ax.grid(axis="x", color="#e1e7ec", linewidth=.7)
        ax.set_axisbelow(True)
        ax.margins(x=.24)
        ax.set_xlabel("MSE reduction relative to the matching AR benchmark (%)", fontsize=10)
        ax.set_title(title, loc="left", fontsize=14, pad=15, weight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        fig.savefig(assets / f"{filename}.png", dpi=180, facecolor="white")
        fig.savefig(assets / f"{filename}.svg", facecolor="white")
        plt.close(fig)


PAGE_CSS = """
:root{--ink:#152c3a;--muted:#526777;--blue:#175d81;--paper:#fff;--bg:#eef2f4;--line:#d5dfe5;--good:#14694e;--bad:#a64035}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:var(--blue);text-underline-offset:3px}a:hover{color:#0d3449}button,input,select{font:inherit}button{cursor:pointer}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #c68130;outline-offset:3px}
.layout{display:grid;grid-template-columns:260px minmax(0,1fr);max-width:1600px;margin:auto}.sidebar{position:sticky;top:0;height:100vh;overflow:auto;padding:30px 24px;background:#152c3a;color:#e8f1f5;font-size:13px}.sidebar .brand{letter-spacing:.13em;font-weight:750;font-size:12px;color:#9fc9dc}.sidebar h2{font-size:16px;color:white;margin:28px 0 10px}.sidebar a{display:block;color:#d4e1e8;text-decoration:none;padding:7px 0;line-height:1.4}.sidebar a:hover{color:white;text-decoration:underline}.sidebar .small{font-size:12px;color:#abc2ce;margin-top:26px}
main{min-width:0;padding:44px 52px 70px}.eyebrow{font-size:12px;font-weight:700;color:var(--blue);text-transform:uppercase;letter-spacing:.13em}h1{font-size:clamp(32px,4vw,50px);line-height:1.12;letter-spacing:-.04em;margin:12px 0 20px;max-width:840px}.lead{font-size:19px;color:var(--muted);max-width:850px}.meta{font-size:13px;color:var(--muted)}.download-links{display:flex;flex-wrap:wrap;gap:16px;margin:22px 0 28px}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:30px 0}.card{background:white;border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:8px;padding:22px}.card .label{font-size:13px;color:var(--muted);font-weight:600}.card .number{font-size:36px;font-weight:700;line-height:1.2;margin:8px 0}.card p{font-size:13px;color:var(--muted);margin:6px 0}.note{padding:18px 22px;border-left:4px solid #bc652b;background:#fff8ef;font-size:15px;margin:24px 0}
.report-section{background:white;border:1px solid var(--line);border-radius:10px;padding:30px 34px;margin:30px 0}.report-section>h2:first-child{margin-top:0}.section-title{font-size:28px;line-height:1.3;letter-spacing:-.025em;margin-bottom:18px}h2{font-size:23px;line-height:1.35;margin:42px 0 14px;letter-spacing:-.02em}h3{font-size:19px;margin:26px 0 10px}p{margin:14px 0}li{margin:7px 0}pre{background:#f0f5f8;border:1px solid #dde6ec;border-radius:7px;padding:16px;overflow:auto;font-size:13px;line-height:1.65}code{font-size:.88em;background:#eef3f6;border-radius:3px;padding:2px 4px}pre code{background:none;padding:0}figure{margin:24px 0}figure img{display:block;width:100%;height:auto;border:1px solid #edf0f3;border-radius:6px}figcaption{font-size:13px;color:var(--muted);margin-top:10px}
.table-wrap{overflow:auto;max-height:580px;border:1px solid var(--line);border-radius:7px;margin:20px 0}.data-table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.45}.data-table th,.data-table td{padding:11px 12px;text-align:left;border-bottom:1px solid #e2e8ed;vertical-align:top}.data-table thead th{position:sticky;top:0;background:#eaf1f5;z-index:1;font-weight:650;white-space:nowrap}.data-table tbody tr:nth-child(even){background:#f9fbfc}.data-table tbody tr:hover{background:#eff6fa}.data-table td{min-width:70px}.data-table td:nth-last-child(-n+3){font-variant-numeric:tabular-nums}.data-table a{overflow-wrap:anywhere}.filter-bar{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:14px;margin:22px 0}.filter-bar label{display:block;font-size:13px;font-weight:600}.filter-bar select,.filter-bar input{display:block;width:100%;border:1px solid #b4c6d1;border-radius:5px;background:white;padding:9px;margin-top:6px}.filter-bar .search{grid-column:span 2}.actions{display:flex;flex-wrap:wrap;align-items:center;gap:12px;font-size:13px;margin:16px 0}.actions button,.detail-toggle{border:1px solid #afc1ce;background:#f0f5f8;color:var(--ink);border-radius:5px;padding:7px 12px}.actions .count{margin-left:auto;font-weight:650}.sort-button{border:0;background:none;color:inherit;padding:0;text-align:left;font-weight:inherit;font-size:inherit}.good{color:var(--good);font-weight:650}.bad{color:var(--bad);font-weight:650}.badge{display:block;width:fit-content;color:#9d5230;background:#fff0e5;font-size:11px;padding:2px 5px;border-radius:3px;margin-top:5px}.detail-toggle{font-size:11px;padding:3px 6px;display:block;margin-top:7px}.detail-row td{background:#eaf2f7!important;font-size:13px;padding:14px!important}.source-link{font-size:12px}.small{font-size:13px;color:var(--muted)}.footer{font-size:13px;color:var(--muted);margin-top:36px}
@media(max-width:1100px){.layout{grid-template-columns:210px minmax(0,1fr)}main{padding:32px 24px}.report-section{padding:24px}.card{padding:16px}.card .number{font-size:30px}.filter-bar{grid-template-columns:repeat(2,minmax(120px,1fr))}}
@media(max-width:760px){.layout{display:block}.sidebar{position:relative;top:auto;height:auto;padding:18px 22px}.sidebar nav{display:none}.sidebar h2,.sidebar .small{display:none}main{padding:24px 14px}.cards{grid-template-columns:1fr}.report-section{padding:20px 16px}h2{font-size:21px}.lead{font-size:17px}.actions .count{margin-left:0}}
@media print{body{background:white;font-size:11px}.layout{display:block}.sidebar,.filter-bar,.actions,.download-links,.detail-toggle{display:none}main{padding:0}.report-section{border:0;padding:0;page-break-before:always}.cards{gap:8px}.card{padding:12px}.card .number{font-size:24px}h1{font-size:32px}h2{break-after:avoid;font-size:18px}pre{white-space:pre-wrap}.table-wrap{max-height:none;overflow:visible;border:0}.data-table{font-size:9px}.data-table th,.data-table td{padding:5px}.data-table thead th{position:static}.note{background:white;border:1px solid #ddd}figure{break-inside:avoid}a{color:inherit}figcaption{font-size:10px}.no-print{display:none}}
"""

PAGE_JS = r"""
const rows = __ROWS__;
const filters = ['family','scale','period','model'];
const tableBody = document.getElementById('results-body');
const count = document.getElementById('row-count');
let sortKey = 'family', sortDirection = 1;
let shown = rows.slice();
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent = value => (value >= 0 ? '+' : '') + (value * 100).toFixed(3) + '%';
for (const name of filters) {
  const select = document.getElementById('filter-' + name);
  for (const value of [...new Set(rows.map(row => row[name]))]) {
    const option = document.createElement('option'); option.value = value; option.textContent = value; select.append(option);
  }
  select.addEventListener('change', render);
}
document.getElementById('search').addEventListener('input', render);
document.getElementById('reset').addEventListener('click', () => {
  filters.forEach(name => document.getElementById('filter-' + name).value = '');
  document.getElementById('search').value = ''; sortKey = 'family'; sortDirection = 1; render();
});
document.querySelectorAll('#explorer .sort-button').forEach(button => button.addEventListener('click', () => {
  const next = button.dataset.key;
  sortDirection = next === sortKey ? -sortDirection : (next === 'improvement' ? -1 : 1);
  sortKey = next; render();
}));
function render() {
  const text = document.getElementById('search').value.toLowerCase();
  shown = rows.filter(row => filters.every(name => !document.getElementById('filter-' + name).value || row[name] === document.getElementById('filter-' + name).value)
    && (!text || [row.family,row.specification,row.alpha_grid,row.source].join(' ').toLowerCase().includes(text)));
  shown.sort((a,b) => sortDirection * (typeof a[sortKey] === 'number' ? a[sortKey] - b[sortKey] : String(a[sortKey]).localeCompare(String(b[sortKey]))));
  tableBody.innerHTML = shown.map((row,index) => `<tr>
    <td>${esc(row.family)}</td><td>${esc(row.specification)}${row.alpha_grid ? '<br><span class="small">'+esc(row.alpha_grid)+'</span>' : ''}${row.diagnostic ? '<span class="badge">Diagnostic OLS</span>' : ''}</td>
    <td>${esc(row.model)}</td><td>${esc(row.scale)}</td><td>${esc(row.period)}</td><td>${row.n}</td>
    <td>${row.MSE.toFixed(6)}</td><td>${row.RMSE.toFixed(6)}</td><td class="${row.improvement > 1e-12 ? 'good' : row.improvement < -1e-12 ? 'bad' : ''}">${percent(row.improvement)}</td>
    <td><a class="source-link" href="../../${esc(row.source)}">CSV source</a><button class="detail-toggle" data-index="${index}" aria-expanded="false">Dates &amp; design</button></td></tr>`).join('');
  if (!shown.length) tableBody.innerHTML = '<tr><td colspan="10">No comparisons match these filters.</td></tr>';
  count.textContent = `${shown.length} of ${rows.length} comparison rows`;
  document.querySelectorAll('#explorer th').forEach(th => th.setAttribute('aria-sort', th.querySelector('button')?.dataset.key === sortKey ? (sortDirection === 1 ? 'ascending' : 'descending') : 'none'));
}
tableBody.addEventListener('click', event => {
  const button = event.target.closest('.detail-toggle'); if (!button) return;
  const tr = button.closest('tr');
  if (button.getAttribute('aria-expanded') === 'true') {tr.nextElementSibling.remove();button.setAttribute('aria-expanded','false');return;}
  const row = shown[Number(button.dataset.index)];
  const detail = document.createElement('tr'); detail.className = 'detail-row';
  detail.innerHTML = `<td colspan="10"><strong>Training:</strong> ${esc(row.window)} · <strong>Extra predictors:</strong> ${row.n_predictors} · <strong>Origins:</strong> ${esc(row.first_origin)} → ${esc(row.last_origin)}<br><strong>Response dates:</strong> ${esc(row.first_target)} → ${esc(row.last_target)} · <strong>Penalty grid:</strong> ${esc(row.alpha_grid || 'See methodology')}<br><strong>Source:</strong> ${esc(row.source)}${row.diagnostic ? '<br>Broad-pool OLS is a diagnostic comparison; consult rank diagnostics before interpreting individual coefficients.' : ''}</td>`;
  tr.after(detail); button.setAttribute('aria-expanded','true');
});
document.getElementById('download-filtered').addEventListener('click', () => {
  const columns = ['family','specification','model','scale','period','window','n_predictors','n','first_origin','last_origin','first_target','last_target','MSE','RMSE','improvement','alpha_grid','diagnostic','source'];
  const csvCell = value => '"' + String(value ?? '').replace(/"/g,'""') + '"';
  const csv = [columns.map(csvCell).join(','), ...shown.map(row => columns.map(column => csvCell(row[column])).join(','))].join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], {type:'text/csv;charset=utf-8'}));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'volatility_results_filtered.csv';anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
render();
"""


def build_page(data, overview, results, generated):
    def toc(doc, prefix):
        return "".join(f'<a href="#{prefix}-{i}">{escape(value[0])}</a>' for i, (kind, value) in enumerate(doc.blocks) if kind == "heading" and value[1] == 2)
    frame = data.frame
    def value(family, specification, scale, period):
        return frame.loc[(frame.family == family) & (frame.specification == specification) & (frame.scale == scale) & (frame.period == period) & (frame.model == "Lasso"), "improvement"].iloc[0] * 100
    cards = [
        ("Monthly forecasting", value("FRED-MD forecast", POOL_LABELS["fredmd_macro"], "level", "Main"), "Main-period MSE reduction", "Macro-only Lasso, levels · 240 months"),
        ("Same-month explanation", value("Same-month explanation", POOL_LABELS["fredmd_full"], "log", "Main"), "Main-period explanatory error reduction", "Full-pool Lasso including VIX, logs · 240 months"),
        ("Quarterly forecasting", value("Quarterly forecast", POOL_LABELS["current_quarterly"], "log", "Main"), "Main-period MSE reduction", "Baseline Lasso, logs · 80 quarters"),
    ]
    card_html = "".join(f'<div class="card"><div class="label">{escape(title)}</div><div class="number">{v:+.2f}%</div><p>{escape(metric)}</p><p>{escape(design)}</p></div>' for title, v, metric, design in cards)
    headers = [("family", "Experiment"), ("specification", "Specification"), ("model", "Model"), ("scale", "Scale"), ("period", "Period"), ("n", "N"), ("MSE", "MSE"), ("RMSE", "RMSE"), ("improvement", "Error reduction"), (None, "Source / design")]
    header_html = "".join(f'<th scope="col"><button class="sort-button" data-key="{key}">{label}</button></th>' if key else f'<th scope="col">{label}</th>' for key, label in headers)
    filters = "".join(f'<label>{label}<select id="filter-{key}"><option value="">All {label.lower()}</option></select></label>' for key, label in [("family", "Experiments"), ("scale", "Scales"), ("period", "Periods"), ("model", "Models")])
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>Market volatility — project methods and results</title><style>{PAGE_CSS}</style></head>
<body><div class="layout"><aside class="sidebar"><div class="brand">VOLATILITY STUDY</div><nav aria-label="Report navigation"><h2>Read the results</h2><a href="#explorer">Explore all comparisons</a><a href="#results">Detailed results</a>{toc(results, 'results')}<h2>Understand the project</h2>{toc(overview, 'overview')}</nav><p class="small">Local, offline report.<br>Based on saved revised-data experiments.<br>No models are refitted when this page opens.</p></aside>
<main><header><div class="eyebrow">Research project · S&amp;P 500 · macroeconomics</div><h1>Market volatility:<br>project methods &amp; results</h1><p class="lead">A detailed account of the data, AR benchmarks, OLS and Lasso experiments—and what their results say about forecasting versus same-month explanation.</p><p class="meta">Generated {escape(generated)} · {len(frame)} verified comparison rows · 8 experiment families</p><div class="download-links"><a href="project_overview.md">Project overview · Markdown</a><a href="results_report.md">Detailed results · Markdown</a><a href="report_manifest.json">Sources &amp; verification manifest</a></div></header>
<div class="cards">{card_html}</div><div class="note"><strong>Different research questions.</strong> The large same-month financial gains use current-period indicators observed ex post. They are explanatory gains, not next-period forecasting gains. Compare each model with its matching AR benchmark, dates and response scale.</div>
<section class="report-section" id="explorer"><h2 class="section-title">Explore every saved comparison</h2><p>Filter the complete result set and click column headings to sort. “Dates &amp; design” shows the actual origins, response dates and training window. Smoke runs are excluded; repeated batch benchmarks are retained as recorded block comparisons.</p><div class="filter-bar">{filters}<label class="search">Search specification, grid or source<input id="search" type="search" placeholder="e.g. VIX, credit, industrial" autocomplete="off"></label></div><div class="actions"><button id="reset">Reset filters</button><button id="download-filtered">Download filtered rows as CSV</button><span class="count" id="row-count" aria-live="polite"></span></div><p class="small">Mixed samples are shown by default. Choose an experiment and scale before ranking error magnitudes. Improvement is AR-relative, not conventional variance explained. Downloaded `improvement` values are fractions; multiply by 100 for percent.</p><div class="table-wrap"><table class="data-table"><thead><tr>{header_html}</tr></thead><tbody id="results-body"></tbody></table></div><noscript>Enable JavaScript for filtering. All results and sources are also shown below and in the Markdown report.</noscript></section>
<article class="report-section" id="results"><h2 class="section-title">Detailed results</h2>{results.html('results')}</article><article class="report-section" id="overview"><h2 class="section-title">Complete project overview</h2>{overview.html('overview')}</article>
<footer class="footer">This report describes the current saved experiments. Forecasting, explanation and statistical inference are distinct. Input hashes and reconciliation counts are in the linked manifest. No third-party scripts, fonts or services are needed to read this page.</footer></main></div>
<script>{PAGE_JS.replace('__ROWS__', json.dumps(data.rows, ensure_ascii=False).replace('</', '<\\/'))}</script></body></html>'''
    return html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    data = ReportData(ROOT)
    data.load_results()
    for path in (ROOT / "src").glob("*.py"):
        data.path(str(path.relative_to(ROOT)))
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    overview = build_overview(data, generated)
    results = build_results(data, generated)
    output = ROOT / "output" / "reports"
    output.mkdir(parents=True, exist_ok=True)
    build_charts(data.frame, output)
    (output / "project_overview.md").write_text(overview.markdown(), encoding="utf-8")
    (output / "results_report.md").write_text(results.markdown(), encoding="utf-8")
    (output / "results.html").write_text(build_page(data, overview, results, generated), encoding="utf-8")
    for relative, digest in data.sources.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Report input changed during generation: {relative}")
    manifest = {"generated_utc": generated, "comparison_rows": len(data.frame),
                "reconciled_metric_rows": data.metric_checks, "smoke_runs_included": False,
                "training_rerun": False, "sources_sha256": data.sources,
                "report_files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                 for name in ["project_overview.md", "results_report.md", "results.html"]},
                "interpretation_note": "Interpretation prose describes this experiment configuration; review it if model definitions or results materially change."}
    (output / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Reconciled {data.metric_checks} comparison rows; preserved {len(data.sources)} source files.")
    print(f"Reports: {output}")


if __name__ == "__main__":
    main()
