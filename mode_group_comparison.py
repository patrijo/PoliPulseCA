# ===================================================================================================================
# mode_group_comparison.py
# Group comparison of sentiment or political tone using transformer models
# in PoliPulseCA. Streamlit UI for uploading, analyzing, and visualizing results.
# Dependencies: torch, transformers, streamlit, nltk, vaderSentiment, matplotlib, seaborn, scipy, pandas
# Author: P. Jost
# Date: May 2025
# License: MIT License (code), CC BY 4.0 (training data/tables/docs) https://creativecommons.org/licenses/by/4.0/
#          Transformer models: Apache 2.0; data: US public domain/open
# ===================================================================================================================

# ───────────────────────────────────────────────────────────────────────────────────────────
# PoliPulse CA - GROUP-LEVEL SENTIMENT/POLITONE TREND ANALYSIS & VISUALISATION
# ───────────────────────────────────────────────────────────────────────────────────────────

import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
from io import BytesIO
import zipfile
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from scipy.stats import spearmanr
from nltk.corpus import stopwords
from transformers import AutoConfig

from shared_utils import (
    TRANSFORMER_PATHS, load_transformer_model, clean_text, split_sentences,
    parse_filename, extract_text, extract_texts_from_zip, vader_sentiment,
    score_transformer_sentences, mark_sig, mann_whitney_between_groups, politone_forward,
    check_uploaded_filenames  # <-- added here
)

# ────── Precompute English stopwords ──────
STOPWORDS = set(stopwords.words("english"))

# ────── Local model directories and discovery ──────
SENTIMENT_MODELS_DIR = "./sentiment_model"
POLITONE_MODELS_DIR  = "./politone_model"
POLITONE_PATHS = {n: n for n in os.listdir(POLITONE_MODELS_DIR)
    if os.path.isdir(os.path.join(POLITONE_MODELS_DIR, n))} if os.path.isdir(POLITONE_MODELS_DIR) else {}

def run():
    # ────── Analysis Mode Selection ──────
    analysis_mode = st.sidebar.selectbox(
        "Analysis Type:", ["Sentiment Analysis", "Political Tone Analysis"]
    )
    is_politone = (analysis_mode == "Political Tone Analysis")
    st.title(f"📊 Group {'Political Tone' if is_politone else 'Sentiment'} Comparison")

    # ────── Device Selection ──────
    st.sidebar.header("💻 Device Settings")
    gpu_available = torch.cuda.is_available()
    device_choice = st.sidebar.radio("Device", ["GPU", "CPU"], index=0 if gpu_available else 0, disabled=not gpu_available)
    if not gpu_available:
        st.sidebar.info("⚠️ GPU not detected; only CPU available")
    device = torch.device("cuda" if device_choice == "GPU" and gpu_available else "cpu")

    # ────── File Upload for Groups ──────
    st.subheader("📁 Upload Speech Files for Each Group")
    group_a_name = st.text_input("Group A Name", value="Group A")
    group_b_name = st.text_input("Group B Name", value="Group B")
    col1, col2 = st.columns(2)

    def load_group(name, container):
        with container:
            st.markdown(f"### {name}")
            uploaded = st.file_uploader(
                f"Upload .txt/.docx or .zip for {name}",
                type=["txt", "docx", "zip"],
                accept_multiple_files=True,
                key=name
            )
            if not uploaded:
                return pd.DataFrame()
            # === Use shared_utils.check_uploaded_filenames ===
            files_to_process, _ = check_uploaded_filenames(uploaded, st_warn=True, context=f"**{name}:**")
            rows = []
            for ftype, fileobj, zname in files_to_process:
                if ftype == "direct":
                    date, speaker = parse_filename(fileobj.name)
                    if date and speaker:
                        text = clean_text(extract_text(fileobj))
                        if not text.strip():
                            continue  # Skip empty or failed extracts!
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
            df_grp = pd.DataFrame(rows)
            if not df_grp.empty:
                df_grp["year"] = pd.to_datetime(df_grp["date"]).dt.year
                df_grp["Group"] = name
            return df_grp

    df_a = load_group(group_a_name, col1)
    df_b = load_group(group_b_name, col2)
    if df_a.empty or df_b.empty:
        st.info("Please upload speeches for both groups to continue.")
        return

    a_min, a_max = int(df_a["year"].min()), int(df_a["year"].max())
    b_min, b_max = int(df_b["year"].min()), int(df_b["year"].max())

    df = pd.concat([df_a, df_b], ignore_index=True)
    df = df.sort_values(by="date").reset_index(drop=True)

    # ────── Method Selection ──────
    st.subheader(f"🧠 {'Political Tone' if is_politone else 'Sentiment'} Method")
    col1, col2 = st.columns([1, 2])

    if not is_politone:
        with col1:
            use_vader = st.checkbox("Use VADER instead of Transformer", value=False)
    else:
        use_vader = False

    transformer_choice = None
    path_map = POLITONE_PATHS if is_politone else TRANSFORMER_PATHS

    if not use_vader:
        with col2:
            transformer_choice = st.selectbox("Choose Transformer Model", list(path_map.keys()))
        model_path = os.path.join(POLITONE_MODELS_DIR if is_politone else SENTIMENT_MODELS_DIR, transformer_choice)
        cfg = AutoConfig.from_pretrained(model_path)
        num_labels = cfg.num_labels
        st.write(f"Model: {transformer_choice} ({num_labels}-class) • Device: {device_choice}")
    else:
        st.write(f"Model: VADER (N/A) • Device: {device_choice}")
        num_labels = None

    method_key = (
        "VADER_Sentiment"
        if (not is_politone and use_vader)
        else f"{transformer_choice}_{'Tone' if is_politone else 'Sentiment'}"
    )
    raw_col = f"Raw_{transformer_choice}_{'Tone' if is_politone else 'Sentiment'}"

    # ────── Processing Options ──────
    with st.expander("⚙️ Processing Options"):
        if not is_politone and use_vader:
            use_temp = use_conf = False
            temp_val = 1.0
            use_wins = False; wins_thr = 2.5
            use_rep = False; rep_thr = 1; rep_pen = 0.5; rep_max = 1.0; exclude_sw = False
        else:
            use_temp = st.checkbox("Use Temperature Scaling", value=False)
            temp_val = st.number_input("Temperature Value", 0.5, 10.0, 1.5, step=0.1) if use_temp else 1.0
            use_conf = st.checkbox("Use Confidence Weighting", value=False)
            use_wins = st.checkbox("Clip Outliers (Winsorize)", value=False)
            wins_thr = st.number_input("Clipping Threshold (±SD)", 0.5, 5.0, 2.5, step=0.1) if use_wins else 2.5
            use_rep = st.checkbox("Use Repetition Penalty", value=False)
            rep_thr = st.number_input("Repetition Threshold", 1, 10, 1, step=1) if use_rep else 1
            rep_pen = st.number_input("Penalty per Word", 0.0, 2.0, 0.5, step=0.1) if use_rep else 0.5
            rep_max = st.number_input("Max Repetition Penalty", 0.0, 2.0, 1.0, step=0.1) if use_rep else 1.0
            exclude_sw = st.checkbox("Exclude stopwords from repetition count", value=False) if use_rep else False

    # ────── Run Analysis ──────
    if st.button("Run Analysis"):
        st.info(
            f"Running **{analysis_mode}** using model "
            f"**{transformer_choice if not use_vader else 'VADER'}** on **{device_choice}**"
        )

        tokenizer = model = None
        if not use_vader:
            st.text(f"🔍 Loading model from: {model_path}")
            tokenizer, model = load_transformer_model(model_path)
            if is_politone:
                politone_forward(model)
            model.to(device)

        results = []
        total = len(df)
        progress = st.progress(0)
        progress_text = st.empty()

        for i, row in enumerate(df.itertuples()):
            progress_text.text(
                f"Processed speech {i+1}/{total}: {row.speaker} "
                f"(Group: {row.Group}, Year: {row.year})"
            )
            sentences = split_sentences(row.text)
            if not use_vader and sentences:
                raw_score = score_transformer_sentences(
                    sentences, tokenizer, model, transformer_choice,
                    device=device
                )
            else:
                raw_score = np.nan

            if not sentences:
                score = 0.0
            elif not is_politone and use_vader:
                score = np.mean([vader_sentiment(s) for s in sentences])
            else:
                score = score_transformer_sentences(
                    sentences, tokenizer, model, transformer_choice,
                    apply_temp=use_temp, temp_value=temp_val,
                    apply_conf=use_conf,
                    apply_wins=use_wins, wins_threshold=wins_thr,
                    apply_repeat=use_rep, repeat_thresh=rep_thr,
                    repeat_penalty_value=rep_pen, repeat_max=rep_max,
                    exclude_stopwords=exclude_sw,
                    device=device
                )
            results.append([row.Group, row.year, row.speaker, raw_score, score])
            progress.progress((i+1)/total)

        progress.empty()
        progress_text.empty()

        # ────── Results DataFrame ──────
        cols = ["Group", "year", "speaker", raw_col, method_key]
        result_df = pd.DataFrame(results, columns=cols)
        result_df["pos"] = result_df.groupby("Group").cumcount()

        # ────── Plot Comparison Chart ──────
        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "figure.dpi": 300,
            "savefig.bbox": "tight"
        })

        group_colors = {
            group_a_name: "#8B1A1A" if is_politone else "#D95F02",
            group_b_name: "#3366CC" if is_politone else "#1B9E77"
        }

        st.subheader(f"📈 {'Political Tone' if is_politone else 'Sentiment'} Comparison Chart")
        fig, ax = plt.subplots(figsize=(5.5, 3))
        pivot = result_df.pivot(index="pos", columns="Group", values=method_key)

        for grp in df["Group"].unique():
            g = result_df[result_df["Group"] == grp]
            color = group_colors[grp]
            sns.lineplot(
                x="pos", y=method_key, data=g,
                label=grp, color=color, ax=ax,
                marker='o', markersize=6, linewidth=2.0, zorder=2
            )
            # Defensive check: only fit if enough data
            xy = g[["pos", method_key]].dropna()
            if len(xy) >= 2 and xy["pos"].nunique() > 1:
                z = np.polyfit(xy["pos"], xy[method_key], 1)
                trend_fn = np.poly1d(z)
                r, _ = spearmanr(g["year"], g[method_key])
                ax.plot(
                    g["pos"], trend_fn(g["pos"]),
                    linestyle=':', color=color,
                    label=f"{grp} Trend (ρ={r:.2f})", zorder=1
                )
            for _, rrow in g.iterrows():
                this_pos = rrow["pos"]
                this_val = rrow[method_key]
                other_grp = [gg for gg in df["Group"].unique() if gg != grp][0]
                other_val = pivot.loc[this_pos, other_grp] if this_pos in pivot.index else this_val
                va, dy = ('bottom', 0.08) if this_val >= other_val else ('top', -0.08)
                year_abbr = f"'{str(rrow['year'])[-2:]}"
                ax.text(
                    this_pos, this_val + dy, year_abbr, fontsize=10, fontweight='bold',
                    ha='center', va=va, alpha=0.6, color=color, zorder=3,
                    path_effects=[path_effects.Stroke(linewidth=3, foreground=(1,1,1,0.8)), path_effects.Normal()]
                )

        ax.fill_between(
            pivot.index, pivot[group_a_name], pivot[group_b_name],
            where=pivot[group_a_name] >= pivot[group_b_name],
            interpolate=True, alpha=0.2, color=group_colors[group_a_name], zorder=0
        )
        ax.fill_between(
            pivot.index, pivot[group_a_name], pivot[group_b_name],
            where=pivot[group_a_name] < pivot[group_b_name],
            interpolate=True, alpha=0.2, color=group_colors[group_b_name], zorder=0
        )
        diff = pivot[group_a_name] - pivot[group_b_name]
        max_idx = diff.abs().idxmax()
        yA, yB = pivot.loc[max_idx, group_a_name], pivot.loc[max_idx, group_b_name]
        mid, gap = ((yA + yB) / 2), (yA - yB)
        ax.vlines(max_idx, ymin=min(yA, yB), ymax=max(yA, yB), color="black", linewidth=1, zorder=1.9)
        ax.text(
            max_idx + 0.1, mid, f"Δ={gap:.2f}",
            ha='left', va='center', fontsize=9.5, fontstyle='italic', zorder=4,
            path_effects=[path_effects.Stroke(linewidth=3, foreground=(1,1,1,0.8)), path_effects.Normal()]
        )
        ax.set_ylabel(f"{'Cooperative-Aggressive (–1 to 1)' if is_politone else 'Sentiment Score (–1 to 1)'}")
        ax.set_xlabel("Speech Index")
        ax.set_ylim(-0.75, 0.75)
        yticks = np.arange(-0.75, 0.76, 0.25)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{y:.2f}'.replace("0.", ".").replace("-0.", "-.") for y in yticks])
        ax.set_xticks(result_df["pos"].unique())
        ax.set_xticklabels([str(i+1) for i in result_df["pos"].unique()])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        legend = ax.legend(loc="lower center", bbox_to_anchor=(0.5,0.00), ncol=2, frameon=False)
        for lh in legend.get_lines():
            lh.set_alpha(1.0)
        for text in legend.get_texts():
            text.set_alpha(1.0)
        st.pyplot(fig)

        # ────── Spearman Correlation ──────
        st.subheader("📊 Spearman Correlation")
        spearman_stats = []
        for grp in df["Group"].unique():
            g = result_df[result_df["Group"] == grp]
            r, p = spearmanr(g["year"], g[method_key])
            spearman_stats.append({
                "Group": grp,
                "Method": method_key.replace(f"_{'Tone' if is_politone else 'Sentiment'}", ""),
                "Spearman ρ": f"{r:.3f}",
                "p-value": p,
                "Significance": mark_sig(p)
            })
        spearman_df = pd.DataFrame(spearman_stats)
        st.dataframe(spearman_df)

        # ────── Mann–Whitney U Test ──────
        st.subheader("📊 Mann-Whitney U Test (Groups)")
        mann_res = mann_whitney_between_groups(result_df, value_column=method_key, group_column="Group")
        mann_df = pd.DataFrame([mann_res])
        st.dataframe(mann_df)

        # ────── Log Options ──────
        options = {
            "Use_VADER": use_vader,
            "Use_Temperature_Scaling": use_temp,
            "Temperature_value": temp_val,
            "Use_Confidence_Weighting": use_conf,
            "Clip_Outliers_Winsorize": use_wins,
            "Winsorize_threshold": wins_thr,
            "Use_Repetition_Penalty": use_rep,
            "Repetition_threshold": rep_thr,
            "Penalty_per_word": rep_pen,
            "Max_repetition_penalty": rep_max,
            "Exclude_stopwords_from_repetition": exclude_sw,
        }
        options_df = pd.DataFrame(list(options.items()), columns=["Option", "Value"])

        # ────── Compute Central Tendencies ──────
        ct_records = []
        for grp, grp_df in result_df.groupby("Group"):
            vals = grp_df[method_key].dropna().values
            if vals.size == 0:
                continue
            q1, q3 = np.percentile(vals, [25, 75])
            mn, mx = np.min(vals), np.max(vals)
            ct_records.append({
                "Group": grp, "mean": np.mean(vals), "median": np.median(vals),
                "std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                "Q1 (25%)": q1, "Q3 (75%)": q3, "IQR (Q3−Q1)": q3 - q1,
                "Min": mn, "Max": mx, "Range (Max−Min)": mx - mn
            })
        ct_df = pd.DataFrame(ct_records)

        # ────── Export Results (ZIP) ──────
        suffix = (
            f"{group_a_name.replace(' ', '_')}_{a_min}-{a_max}"
            f"_vs_{group_b_name.replace(' ', '_')}_{b_min}-{b_max}"
            f"_{'tone' if is_politone else 'sentiment'}_trend_comparison"
        )
        excel_buf = BytesIO()
        with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
            result_df.to_excel(writer, index=False, sheet_name="Results")
            spearman_df.to_excel(writer, index=False, sheet_name="Spearman")
            mann_df.to_excel(writer, index=False, sheet_name="Mann-Whitney")
            options_df.to_excel(writer, index=False, sheet_name="Options")
            ct_df.to_excel(writer, index=False, sheet_name="Central_Tendencies")
        svg_buf = BytesIO(); fig.savefig(svg_buf, format="svg", bbox_inches="tight")
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr(f"{suffix}.xlsx", excel_buf.getvalue())
            zf.writestr(f"{suffix}.svg", svg_buf.getvalue())
        zip_buf.seek(0)
        st.download_button(
            "📥 Download Results (ZIP)",
            data=zip_buf.getvalue(),
            file_name=f"{suffix}.zip",
            mime="application/zip"
        )

if __name__ == "__main__":
    run()
