# ===================================================================================================================
# mode_general.py
# General mode for sentiment or political tone trend analysis using transformer models
# in PoliPulseCA. Streamlit UI for uploading, filtering, analyzing, and visualizing results.
# Dependencies: torch, transformers, streamlit, nltk, vaderSentiment, matplotlib, seaborn, scipy, pandas
# Author: 
# Date: May 2025
# License: MIT License (code), CC BY 4.0 (training data/tables/docs) https://creativecommons.org/licenses/by/4.0/
#          Transformer models: Apache 2.0; data: US public domain/open
# ===================================================================================================================

# ───────────────────────────────────────────────────────────────────────────────────────────
# PoliPulse CA - GENERAL SENTIMENT/POLITONE TREND ANALYSIS & VISUALISATION
# ───────────────────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import time
import torch
from transformers import AutoConfig
from scipy.stats import spearmanr, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from io import BytesIO
import zipfile
import matplotlib.pyplot as plt

from shared_utils import (
    TRANSFORMER_PATHS, load_transformer_model, clean_text, split_sentences, parse_filename,
    extract_text, extract_texts_from_zip, vader_sentiment, textblob_sentiment,
    score_transformer_sentences, mark_sig, compute_mw_effect_size,
    visualize_general_trend, politone_forward,
    check_uploaded_filenames  # <-- added here
)

# ────── Model Paths ──────
SENTIMENT_MODELS_DIR = "./sentiment_model"
POLITONE_MODELS_DIR  = "./politone_model"
POLITONE_PATHS = {n: n for n in os.listdir(POLITONE_MODELS_DIR)
    if os.path.isdir(os.path.join(POLITONE_MODELS_DIR, n))} if os.path.isdir(POLITONE_MODELS_DIR) else {}

def run():
    # ────── Analysis Type ──────
    mode = st.sidebar.selectbox("Analysis Type", ["Sentiment Analysis", "Political Tone Analysis"])
    is_politone = (mode == "Political Tone Analysis")
    st.title("📊 Political Tone Trend Analysis – General Mode" if is_politone else "📊 Sentiment Trend Analysis – General Mode")

    # ────── Device Selection ──────
    st.sidebar.header("💻 Device Settings")
    use_gpu = torch.cuda.is_available() and st.sidebar.radio("Processing Device", ["GPU", "CPU"], index=0) == "GPU"
    if not torch.cuda.is_available():
        st.sidebar.radio("Processing Device", ["CPU"], index=0, disabled=True)
    device = torch.device("cuda" if use_gpu else "cpu")

    # ────── Upload & Parse Speech Files ──────
    uploads = st.file_uploader("📁 Upload speech files", type=["txt", "docx", "zip"], accept_multiple_files=True)
    if not uploads:
        st.info("⬆️ Upload speech files to begin.")
        return

    # === Use shared_utils.check_uploaded_filenames ===
    files_to_process, _ = check_uploaded_filenames(uploads, st_warn=True, context="**Uploaded files:**")

    rows = []
    for ftype, fileobj, zname in files_to_process:
        if ftype == "direct":
            date, speaker = parse_filename(fileobj.name)
            if date and speaker:
                text = clean_text(extract_text(fileobj))
                if not text.strip():
                    continue
                rows.append({"date": date, "speaker": speaker, "text": text})
        elif ftype == "zip":
            with zipfile.ZipFile(fileobj, "r") as z:
                with z.open(zname) as f:
                    date, speaker = parse_filename(os.path.basename(zname))
                    if date and speaker:
                        if zname.endswith(".txt"):
                            txt = f.read().decode("utf-8")
                        elif zname.endswith(".docx"):
                            import io
                            from docx import Document
                            txt = "\n".join(p.text for p in Document(io.BytesIO(f.read())).paragraphs)
                        else:
                            txt = ""
                        text = clean_text(txt)
                        if not text.strip():
                            continue
                        rows.append({"date": date, "speaker": speaker, "text": text})
    df = pd.DataFrame(rows)

    if df.empty:
        st.warning("No valid speeches found after filtering. Please check filenames and file contents.")
        return

    df["year"] = pd.to_datetime(df["date"]).dt.year.astype(str)
    df["Select"] = False
    df = df.sort_values("date").reset_index(drop=True)

    # ────── Filtering ──────
    st.sidebar.header("Filter Options")
    years = sorted(df["year"].unique())
    speakers = sorted(df["speaker"].unique())
    sel_years = st.sidebar.multiselect("Years", years, default=years)
    sel_spkrs = st.sidebar.multiselect("Speakers", speakers, default=speakers)
    df = df[df["year"].isin(sel_years) & df["speaker"].isin(sel_spkrs)]

    # ────── Speech Selection Table ───────────
    st.subheader("📄 Select Speeches to Analyze")
    if st.checkbox("Select All"): df["Select"] = True
    edited = st.data_editor(df[["Select", "year", "speaker", "text"]], use_container_width=True, hide_index=True)
    df["Select"] = edited["Select"]
    selected_df = df[df["Select"]].sort_values("date").reset_index(drop=True)

    # ────── Transformer Model Selection ──────
    st.subheader("🧠 Transformer Model & Device")
    if is_politone:
        model_name = st.selectbox("Transformer Model", list(POLITONE_PATHS.keys()))
        model_path = Path(POLITONE_MODELS_DIR) / POLITONE_PATHS[model_name]
    else:
        model_name = st.selectbox("Transformer Model", list(TRANSFORMER_PATHS.keys()))
        model_path = Path(SENTIMENT_MODELS_DIR) / TRANSFORMER_PATHS[model_name]
    model_path = model_path.as_posix()
    cfg = AutoConfig.from_pretrained(model_path)
    num_labels = cfg.num_labels
    st.write(f"**Model:** {model_name} ({num_labels}-class) • **Device:** {'GPU' if use_gpu else 'CPU'}")

    # ────── Options ──────
    temp_val, clip_thr = 1.0, 2.5
    rpt_thr, rpt_pen, rpt_max = 1, 0.5, 1.0
    exclude_sw = False
    with st.expander(f"⚙️ {'Political Tone' if is_politone else 'Sentiment'} Options"):
        apply_temp = st.checkbox("Use Temperature Scaling")
        if apply_temp:
            temp_val = st.number_input("Temperature (T)", 0.5, 5.0, 1.5)
        apply_conf = st.checkbox("Use Confidence Weighting")
        apply_wins = st.checkbox("Clip Outliers (Winsorize)")
        if apply_wins:
            clip_thr = st.number_input("Winsorizing Threshold (SD)", 0.5, 5.0, 2.5)
        apply_rpt = st.checkbox("Use Repetition Penalty")
        if apply_rpt:
            rpt_thr = st.number_input("Repetition Threshold", 1, 10, 1)
            rpt_pen = st.number_input("Penalty per Word", 0.0, 2.0, 0.5)
            rpt_max = st.number_input("Max Penalty", 0.0, 2.0, 1.0)
            exclude_sw = st.checkbox("Exclude stopwords from count")

    # ────── Run Analysis ──────
    st.subheader(f"🔍 {'Political Tone' if is_politone else 'Sentiment'} Analysis")
    if st.button("Run Analysis"):
        if selected_df.empty:
            st.warning("Please select at least one speech.")
            return

        tokenizer, model = load_transformer_model(model_path)
        if is_politone:
            politone_forward(model)
        model.to(device)
        st.info(f"Running **{model_name}** on **{'GPU' if use_gpu else 'CPU'}**")

        records, prog, status = [], st.progress(0), st.empty()
        total = len(selected_df)

        for i, row in enumerate(selected_df.itertuples(), 1):
            status.text(f"Processing {i}/{total}: {row.speaker} ({row.year})")
            sents = split_sentences(row.text)
            if not is_politone:
                t0 = time.perf_counter()
                vd = np.mean([vader_sentiment(s) for s in sents]) if sents else 0.0
                t_vd = time.perf_counter() - t0

                t0 = time.perf_counter()
                tb = np.mean([textblob_sentiment(s) for s in sents]) if sents else 0.0
                t_tb = time.perf_counter() - t0

            t0 = time.perf_counter()
            proc_tf = score_transformer_sentences(
                sents, tokenizer, model, model_name,
                apply_temp=apply_temp, temp_value=temp_val,
                apply_conf=apply_conf,
                apply_repeat=apply_rpt, repeat_thresh=rpt_thr,
                repeat_penalty_value=rpt_pen, repeat_max=rpt_max,
                exclude_stopwords=exclude_sw,
                apply_wins=apply_wins, wins_threshold=clip_thr,
                device=device
            )
            t_tf = time.perf_counter() - t0
            raw_tf = score_transformer_sentences(sents, tokenizer, model, model_name, device=device)

            if not is_politone:
                records.append([row.year, row.speaker, vd, tb, raw_tf, proc_tf, t_vd, t_tb, t_tf])
            else:
                records.append([row.year, row.speaker, raw_tf, proc_tf, t_tf])
            prog.progress(i / total)
        prog.empty(), status.empty()

        # ────── Output Table ──────
        if not is_politone:
            cols = ["year", "speaker", "VADER_Sentiment", "TextBlob_Sentiment",
                    "Raw_Transformer_Sentiment", "Transformer_Sentiment",
                    "VADER_Time", "TextBlob_Time", "Transformer_Time"]
        else:
            cols = ["year", "speaker", "Raw_Political_Tone", "Political_Tone", "Transformer_Time"]
        out_df = pd.DataFrame(records, columns=cols)
        st.success("✅ Analysis complete!")
        st.dataframe(out_df)

        # ────── Central Tendencies ──────
        metric_col = "Political_Tone" if is_politone else "Transformer_Sentiment"
        ct_records = []
        for speaker, grp in out_df.groupby("speaker")[metric_col]:
            vals = grp.values
            q1, q3 = np.percentile(vals, [25, 75])
            mn, mx = np.min(vals), np.max(vals)
            ct_records.append({
                "speaker": speaker, "mean": np.mean(vals), "median": np.median(vals),
                "std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                "Q1 (25%)": q1, "Q3 (75%)": q3, "IQR (Q3−Q1)": q3 - q1,
                "Min": mn, "Max": mx, "Range (Max−Min)": mx - mn
            })
        ct_df = pd.DataFrame(ct_records)

        # ────── Options Logging ──────
        options = {
            "Use_Temperature_Scaling": apply_temp,
            "Temperature_value": temp_val,
            "Use_Confidence_Weighting": apply_conf,
            "Clip_Outliers_Winsorize": apply_wins,
            "Winsorize_threshold": clip_thr,
            "Use_Repetition_Penalty": apply_rpt,
            "Repetition_threshold": rpt_thr,
            "Penalty_per_word": rpt_pen,
            "Max_repetition_penalty": rpt_max,
            "Exclude_stopwords_from_repetition": exclude_sw,
        }
        options_df = pd.DataFrame(list(options.items()), columns=["Option", "Value"])

        # ────── Visualization & Statistics ──────
        if is_politone:
            fig, ax = plt.subplots()
            yrs = out_df["year"].astype(int)
            vals = out_df["Political_Tone"].astype(float)
            ax.plot(yrs, vals, label="Political_Tone", zorder=2)
            r, p = spearmanr(yrs, vals)
            z = np.polyfit(yrs, vals, 1)
            ax.plot(yrs, np.poly1d(z)(yrs), linestyle=':', label=f"Trend (ρ={r:.2f})", zorder=1)
            ax.set_title("Political Tone Trend")
            ax.set_xlabel("Year")
            ax.set_ylabel("Cooperative–Aggressive (-1 to 1)")
            ax.set_ylim(-1, 1)
            ax.legend()
            min_year, max_year = yrs.min(), yrs.max()
            span = max_year - min_year
            step = 1 if span <= 10 else (5 if span <= 20 else 10)
            xticks = np.arange(min_year, max_year + 1, step)
            ax.set_xticks(xticks); ax.set_xticklabels(xticks)
            ax.grid(True)
            for year in range(min_year, max_year + 1):
                ax.axvline(x=year, color="gray", linestyle="--", alpha=0.2, zorder=0)
            st.pyplot(fig)
            trend_df = pd.DataFrame([{"Method": "Political_Tone", "Spearman ρ": f"{r:.3f}", "p-value": p, "Significance": mark_sig(p)}])
        else:
            fig, trend_df = visualize_general_trend(out_df)

        st.subheader("📊 Spearman Correlation")
        st.dataframe(trend_df)

        # ────── Kruskal–Wallis & Mann–Whitney U (Sentiment only) ──────
        if not is_politone:
            k, p = kruskal(out_df["VADER_Sentiment"], out_df["TextBlob_Sentiment"], out_df["Transformer_Sentiment"])
            eta2 = (k - 2) / (len(out_df) * 3 - 3)
            kw_df = pd.DataFrame([{"Statistic": k, "p-value": p, "Effect Size (η²)": f"{eta2:.3f}", "Significance": mark_sig(p)}])
            st.subheader("📊 Kruskal–Wallis Test")
            st.dataframe(kw_df)

            mw_rows, pvals = [], []
            for a, b in [("VADER_Sentiment", "TextBlob_Sentiment"),
                         ("VADER_Sentiment", "Transformer_Sentiment"),
                         ("TextBlob_Sentiment", "Transformer_Sentiment")]:
                u, p0 = mannwhitneyu(out_df[a], out_df[b])
                r_es = compute_mw_effect_size(u, len(out_df), len(out_df))
                mw_rows.append({"Comparison": f"{a} vs {b}", "U-statistic": u, "p-value": p0, "Effect Size (r)": f"{r_es:.3f}"})
                pvals.append(p0)
            _, adj, _, _ = multipletests(pvals, method="holm")
            for idx, row in enumerate(mw_rows):
                row["Holm-adjusted p"] = adj[idx]
                row["Significance"] = mark_sig(adj[idx])
            mw_df = pd.DataFrame(mw_rows)
            st.subheader("📊 Mann–Whitney U Test")
            st.dataframe(mw_df)

        # ────── Save & Download ──────
        first, last = out_df["year"].min(), out_df["year"].max()
        prefix = (f"Political_Tone_Trend_{first}-{last}_{model_name}" if is_politone
                  else f"Sentiment_Trend_{first}-{last}_{model_name}")

        excel_buf = BytesIO()
        with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
            out_df.to_excel(writer, sheet_name="Results", index=False)
            trend_df.to_excel(writer, sheet_name="Spearman", index=False)
            if not is_politone:
                kw_df.to_excel(writer, sheet_name="Kruskal-Wallis", index=False)
                mw_df.to_excel(writer, sheet_name="Mann-Whitney", index=False)
            options_df.to_excel(writer, sheet_name="Options", index=False)
            ct_df.to_excel(writer, sheet_name="Central_Tendencies", index=False)
        svg_buf = BytesIO(); fig.savefig(svg_buf, format="svg", bbox_inches="tight")
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr(f"{prefix}.xlsx", excel_buf.getvalue())
            zf.writestr(f"{prefix}.svg", svg_buf.getvalue())
        zip_buf.seek(0)
        zip_name = f"results_{first}-{last}_{model_name}.zip"
        st.download_button("📥 Download All Results (ZIP)", data=zip_buf.getvalue(), file_name=zip_name, mime="application/zip")

if __name__ == "__main__":
    run()
