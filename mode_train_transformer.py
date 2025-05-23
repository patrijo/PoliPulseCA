# ===================================================================================================================
# mode_finetune.py
# Transformer Model Fine-Tuning & Evaluation App for Political Tone Classification (PoliPulseCA)
# Streamlit UI for configuring, training, evaluating, and exporting DistilBERT models as three-class classifiers
# for political tone (Cooperative / Neutral / Aggressive) on sentence-level text data.
# Dependencies: torch, transformers, streamlit, nltk, matplotlib, pandas, sklearn, tqdm
# Author: P. Jost
# Date: May 2025
# License: MIT License (code), CC BY 4.0 (training data/tables/docs) https://creativecommons.org/licenses/by/4.0/
#          Transformer models: Apache 2.0; data: US public domain/open
# ===================================================================================================================

# ───────────────────────────────────────────────────────────────────────────────────────────
# PoliPulse CA - MODEL TRAINING & FINE-TUNING APP (DistilBERT, three-class politone)
# ───────────────────────────────────────────────────────────────────────────────────────────

import os
import random
import numpy as np
import pandas as pd
import streamlit as st
from tqdm import tqdm
import datetime
from nltk.tokenize import sent_tokenize
import re

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertConfig,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
torch.classes.__path__ = []  # fix torch + streamlit observer issue

def run():
    # =============================================================================
    # Utility functions
    # =============================================================================
    def apply_temperature_scaling(logits, temperature):
        exp_logits = np.exp(logits / temperature)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def compute_weighted_score(softmax_probs):
        return -1 * softmax_probs[0] + 1 * softmax_probs[2]

    def windsorise(scores, pct):
        low, high = np.percentile(scores, pct), np.percentile(scores, 100-pct)
        return np.clip(scores, low, high)

    def repeated_words_compensation(text, score, thr, penalty, mx):
        counts = {}
        for w in text.split():
            counts[w] = counts.get(w, 0) + 1
        p = sum(min(penalty*(c-thr), mx) for c in counts.values() if c>thr)
        return score - p

    # =============================================================================
    # Focal Loss
    # =============================================================================
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, weight=None):
            super().__init__()
            self.gamma = gamma
            self.weight = weight
        def forward(self, inputs, targets):
            ce = nn.functional.cross_entropy(
                inputs, targets, weight=self.weight, reduction='none'
            )
            pt = torch.exp(-ce)
            return ((1-pt)**self.gamma * ce).mean()

    # =============================================================================
    # Dataset for fine-tuning
    # =============================================================================
    class SpeechDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128,
                    augment_mask=False, mask_prob=0.0,
                    augment_shuffle=False, shuffle_dist=3):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.augment_mask = augment_mask
            self.mask_prob = mask_prob
            self.augment_shuffle = augment_shuffle
            self.shuffle_dist = shuffle_dist
            self.mask_token_id = tokenizer.mask_token_id

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            enc = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].squeeze(0)
            attention_mask = enc['attention_mask'].squeeze(0)

            # augmentation: random masking and token shuffling
            if self.augment_mask:
                ids_list = input_ids.tolist()
                for i in range(len(ids_list)):
                    if random.random() < self.mask_prob:
                        ids_list[i] = self.mask_token_id
                input_ids = torch.tensor(ids_list, dtype=torch.long)
            if self.augment_shuffle:
                ids_list = input_ids.tolist()
                for i in range(len(ids_list)):
                    swap_i = min(len(ids_list)-1,
                                max(0, i + random.randint(-self.shuffle_dist, self.shuffle_dist)))
                    ids_list[i], ids_list[swap_i] = ids_list[swap_i], ids_list[i]
                input_ids = torch.tensor(ids_list, dtype=torch.long)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

    # =============================================================================
    # Model wrapper for three-class politone classification
    # =============================================================================
    class ThreeHeadClassifier(nn.Module):
        def __init__(self, base_model, hidden_size,
                    frozen_head=None, dropout_rate=0.3,
                    enable_learnable_temp=False):
            super().__init__()
            self.base_model = base_model
            if frozen_head is not None:
                for p in frozen_head.parameters():
                    p.requires_grad = False
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(hidden_size, 3)
            self.temp = nn.Parameter(torch.ones(1)) if enable_learnable_temp else None

        def forward(self, input_ids, attention_mask):
            out = self.base_model.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            cls = out.last_hidden_state[:,0]
            h = self.dropout(cls)
            logits = self.classifier(h)
            return logits / self.temp if self.temp is not None else logits

    # =============================================================================
    # Streamlit UI: Model Training & Fine-Tuning Interface
    # =============================================================================

    st.title("PoliPulse CA Model Training & Evaluation")

    # -------------------------------------------------------------------------
    # 0. Master log: display previous runs
    # -------------------------------------------------------------------------
    master_log_path = "training_log.xlsx"
    if os.path.exists(master_log_path):
        st.subheader("Master Log of Training Runs")
        st.dataframe(pd.read_excel(master_log_path))

    # -------------------------------------------------------------------------
    # 1. Training data
    # -------------------------------------------------------------------------
    st.sidebar.header("1. Training Data Source")
    training_data_file = "trainingdata/sentences.xlsx"
    if not os.path.exists(training_data_file):
        st.sidebar.error("Training data file not found!")
    else:
        st.sidebar.write(f"Using: {training_data_file}")

    # -------------------------------------------------------------------------
    # 2. Model selection
    # -------------------------------------------------------------------------
    st.sidebar.header("2. Select Offline Model")
    model_folder = "offline-models"
    models = (
        [d for d in os.listdir(model_folder)
        if os.path.isdir(os.path.join(model_folder, d))]
        if os.path.isdir(model_folder) else []
    )
    selected_model_name = (
        st.sidebar.selectbox("Select a model", models)
        if models else None
    )

    # -------------------------------------------------------------------------
    # 3. Training parameters
    # -------------------------------------------------------------------------
    st.sidebar.header("3. Training Parameters")
    epochs = st.sidebar.number_input("Epochs", 1, 50, value=4)
    batch_size = st.sidebar.number_input("Batch Size", 1, 64, value=8)
    enable_discriminative = st.sidebar.checkbox(
        "Enable Discriminative Learning Rates", value=False
    )
    if not enable_discriminative:
        learning_rate = st.sidebar.number_input(
            "Learning Rate", value=3e-5, format="%.5e"
        )
    else:
        head_lr = st.sidebar.number_input(
            "Classifier Head Learning Rate", value=3e-4, format="%.5e"
        )
        top_lr  = st.sidebar.number_input(
            "Top Layers Learning Rate",      value=3e-5, format="%.5e"
        )
    max_length = st.sidebar.number_input(
        "Max Sequence Length", 64, 512, value=128
    )
    enable_learnable_temp = st.sidebar.checkbox(
        "Enable Learnable Temperature", value=False
    )

    # -------------------------------------------------------------------------
    # 4. Scheduler
    # -------------------------------------------------------------------------
    st.sidebar.header("4. LR Scheduler")
    sched_type = st.sidebar.selectbox(
        "Scheduler", [
            "None",
            "LinearWarmup",
            "OneCycle",
            "CosineRestarts",
            "ReduceOnPlateau"
        ]
    )
    warmup_proportion = None
    max_lr_one = None; pct_start=None
    T0=None; Tmult=None
    reduce_fac=None; reduce_pat=None

    if sched_type=="LinearWarmup":
        warmup_proportion = st.sidebar.number_input(
            "Warmup Proportion", 0.0, 1.0, value=0.1, step=0.01
        )
    elif sched_type=="OneCycle":
        max_lr_one = st.sidebar.number_input(
            "OneCycle max_lr",
            value=(learning_rate if not enable_discriminative else top_lr),
            format="%.5e"
        )
        pct_start = st.sidebar.number_input(
            "OneCycle pct_start", 0.0, 1.0, value=0.3, step=0.01
        )
    elif sched_type=="CosineRestarts":
        T0 = st.sidebar.number_input("T₀", 1, 100, value=10)
        Tmult = st.sidebar.number_input("Tᵐᵘˡᵗ", 1, 10, value=1)
    elif sched_type=="ReduceOnPlateau":
        reduce_fac = st.sidebar.number_input(
            "Reduce factor", 0.0, 1.0, value=0.5, step=0.05
        )
        reduce_pat = st.sidebar.number_input(
            "Reduce patience", 0, 10, value=1
        )

    # -------------------------------------------------------------------------
    # 5. Loss function & class weights
    # -------------------------------------------------------------------------
    st.sidebar.header("5. Loss Function")
    loss_fn = st.sidebar.selectbox("Choose Loss", ["CrossEntropy","Focal"])
    coop_w = st.sidebar.number_input("Cooperative Weight", value=1.0)
    neutral_w = st.sidebar.number_input("Neutral Weight", value=1.0)
    aggr_w = st.sidebar.number_input("Aggressive Weight", value=1.0)
    focal_gamma = (
        st.sidebar.number_input("Focal γ", 0.0, 5.0, value=2.0, step=0.1)
        if loss_fn=="Focal" else None
    )

    # -------------------------------------------------------------------------
    # 6. Regularization
    # -------------------------------------------------------------------------
    st.sidebar.header("6. Regularization Parameters")
    dropout_rate = st.sidebar.number_input("Dropout Rate", value=0.3, step=0.05)
    embedding_dropout = st.sidebar.number_input("Embedding Dropout", value=0.0, step=0.05)
    label_smoothing = st.sidebar.number_input("Label Smoothing", value=0.0, step=0.01)
    weight_decay = st.sidebar.number_input("Weight Decay", value=0.0, step=0.005)
    early_stop_pat = st.sidebar.number_input("Early Stopping Patience", 1, 10, value=2)

    # -------------------------------------------------------------------------
    # 7. Partial unfreezing
    # -------------------------------------------------------------------------
    st.sidebar.header("7. Partial Unfreezing")
    num_unfreeze = st.sidebar.number_input(
        "Unfreeze top N layers", 0, 6, value=2
    )

    # -------------------------------------------------------------------------
    # 8. Self augmentation
    # -------------------------------------------------------------------------
    st.sidebar.header("8. Self Augmentation")
    use_masking = st.sidebar.checkbox("Enable Random Masking", value=False)
    mask_prob = st.sidebar.number_input(
        "Masking Probability", 0.0, 1.0, value=0.15, step=0.05
    ) if use_masking else 0.0
    use_shuffle = st.sidebar.checkbox("Enable Token Shuffling", value=False)
    shuffle_dist = st.sidebar.number_input(
        "Shuffle Window Size", 1, 10, value=3, step=1
    ) if use_shuffle else 0

    # -------------------------------------------------------------------------
    # 9. Advanced Evaluation Options (inference & scoring post-processing)
    # -------------------------------------------------------------------------
    st.sidebar.header("9. Advanced Evaluation Options")
    use_temp_scaling = st.sidebar.checkbox("Enable Inference Temp Scaling", value=False)
    default_temp = st.session_state.get("learned_temp", 1.0)
    if use_temp_scaling:
        temperature_scaling = st.sidebar.slider(
            "Temperature (inf)", 1, 5, int(round(default_temp)), step=1
        )
    else:
        temperature_scaling = 1.0
    use_windsorising = st.sidebar.checkbox("Enable Windsorising", value=False)
    windsorise_threshold = st.sidebar.number_input("Percentile", 0.0, 50.0, value=0.0) if use_windsorising else 0.0
    use_repeat_comp = st.sidebar.checkbox("Enable Repeat Compensation", value=False)
    if use_repeat_comp:
        repeat_thr = st.sidebar.number_input("Repeat Threshold", 1, 10, value=1)
        repeat_pen = st.sidebar.number_input("Repeat Penalty", value=0.5)
        repeat_max = st.sidebar.number_input("Repeat Max Penalty", value=1.0)
    else:
        repeat_thr = 10; repeat_pen = 0.0; repeat_max = 0.0
    use_conf_weighting = st.sidebar.checkbox("Enable Confidence Weighting", value=False)

    # =============================================================================
    # Training process and evaluation logic
    # =============================================================================
    if st.sidebar.button("Start Training"):

        # --------------------------------------------------------------
        # Validate input selections
        # --------------------------------------------------------------
        if not os.path.exists(training_data_file):
            st.error("Data file missing!"); st.stop()
        if selected_model_name is None:
            st.error("Select a model!"); st.stop()

        # --------------------------------------------------------------
        # Load and split data
        # --------------------------------------------------------------
        df = pd.read_excel(training_data_file)
        df.columns = df.columns.str.strip().str.lower()
        df['sentence'] = df['sentence'].fillna("").astype(str)
        if not {'sentence','label'}.issubset(df.columns):
            st.error("Need columns 'sentence' and 'label'."); st.stop()

        texts = df['sentence'].tolist()
        raw   = df['label'].tolist()
        mapping = {-1:0,0:1,1:2}
        labels  = [mapping[x] for x in raw]

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.3,
            stratify=labels, random_state=99
        )
        pd.DataFrame({'sentence':train_texts,'label':train_labels}).to_excel("trainingsentences.xlsx",index=False)
        pd.DataFrame({'sentence':val_texts,'label':val_labels}).to_excel("validationsentences.xlsx",index=False)
        st.success("Train/val splits saved.")

        # --------------------------------------------------------------
        # Load model & tokenizer, freeze base classifier head, dropout
        # --------------------------------------------------------------
        base_path = os.path.join(model_folder, selected_model_name)
        tokenizer = DistilBertTokenizerFast.from_pretrained(base_path)
        base_model = DistilBertForSequenceClassification.from_pretrained(base_path)

        # Freeze original classifier head
        for p in base_model.classifier.parameters():
            p.requires_grad = False
        frozen_head = base_model.classifier

        # Embedding dropout
        base_model.distilbert.embeddings.dropout = nn.Dropout(embedding_dropout)

        # Partial unfreezing (top N layers)
        total_layers = 6
        threshold = total_layers - num_unfreeze
        for n,p in base_model.distilbert.named_parameters():
            if "layer." in n:
                idx = int(n.split("layer.")[1].split(".")[0])
                p.requires_grad = (idx >= threshold)
            else:
                p.requires_grad = False

        # Model wrapper
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ThreeHeadClassifier(
            base_model,
            base_model.config.dim,
            frozen_head=frozen_head,
            dropout_rate=dropout_rate,
            enable_learnable_temp=enable_learnable_temp
        ).to(device)
        base_model = base_model.to(device)
        base_model.eval()
        st.write(f"🖥️ Running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        # Show learned temperature if enabled
        if enable_learnable_temp and model.temp is not None:
            learned_temp = model.temp.item()
            st.session_state["learned_temp"] = learned_temp
            st.sidebar.markdown(f"**Learned Temperature:** {learned_temp:.4f}")

        # --------------------------------------------------------------
        # Prepare data loaders
        # --------------------------------------------------------------
        train_ds = SpeechDataset(
            train_texts, train_labels, tokenizer, max_length,
            augment_mask=use_masking, mask_prob=mask_prob,
            augment_shuffle=use_shuffle, shuffle_dist=shuffle_dist
        )
        val_ds = SpeechDataset(val_texts, val_labels, tokenizer, max_length)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # --------------------------------------------------------------
        # Optimizer & scheduler setup
        # --------------------------------------------------------------
        class_wts = torch.tensor([coop_w, neutral_w, aggr_w]).to(device)
        if enable_discriminative:
            head_params = list(model.classifier.parameters())
            top_params = [p for n,p in model.base_model.distilbert.named_parameters() if p.requires_grad]
            optimizer = optim.AdamW([
                {'params': head_params, 'lr': head_lr},
                {'params': top_params,  'lr': top_lr}
            ], weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate, weight_decay=weight_decay
            )

        if sched_type=="LinearWarmup":
            total_steps = len(train_loader)*epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(warmup_proportion*total_steps),
                num_training_steps=total_steps
            )
        elif sched_type=="OneCycle":
            total_steps = len(train_loader)*epochs
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr_one,
                total_steps=total_steps, pct_start=pct_start, cycle_momentum=False
            )
        elif sched_type=="CosineRestarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T0, T_mult=Tmult
            )
        elif sched_type=="ReduceOnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',
                factor=reduce_fac, patience=reduce_pat
            )
        else:
            scheduler = None

        # --------------------------------------------------------------
        # Training loop
        # --------------------------------------------------------------
        prog = st.progress(0)
        total_batches = len(train_loader)*epochs
        done = 0
        best_val = float('inf')
        no_imp = 0
        train_hist, val_hist = [], []
        best_state = None

        for ep in range(epochs):
            model.train()
            ep_loss = 0.0
            st.write(f"Epoch {ep+1}/{epochs}")

            for batch in train_loader:
                optimizer.zero_grad()
                ids = batch['input_ids'].to(device)
                msk = batch['attention_mask'].to(device)
                lbl = batch['labels'].to(device)

                logits = model(ids, msk)
                if loss_fn=="CrossEntropy":
                    crit = nn.CrossEntropyLoss(weight=class_wts, label_smoothing=label_smoothing)
                    loss = crit(logits, lbl)
                else:
                    loss = FocalLoss(gamma=focal_gamma, weight=class_wts)(logits, lbl)

                loss.backward()
                optimizer.step()
                if scheduler and sched_type in ["LinearWarmup","OneCycle","CosineRestarts"]:
                    scheduler.step()

                ep_loss += loss.item()
                done += 1
                prog.progress(done/total_batches)

            avg_tr = ep_loss / len(train_loader)
            train_hist.append(avg_tr)
            st.write(f"Training Loss: {avg_tr:.4f}")

            # Validation step
            model.eval()
            vloss = 0.0
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                msk = batch['attention_mask'].to(device)
                lbl = batch['labels'].to(device)
                with torch.no_grad():
                    lg = model(ids, msk)
                    if loss_fn=="CrossEntropy":
                        lcrit = nn.CrossEntropyLoss(weight=class_wts, label_smoothing=label_smoothing)
                        l = lcrit(lg, lbl).item()
                    else:
                        l = FocalLoss(gamma=focal_gamma, weight=class_wts)(lg, lbl).item()
                vloss += l

            avg_val = vloss / len(val_loader)
            val_hist.append(avg_val)
            st.write(f"Validation Loss: {avg_val:.4f}")

            if scheduler and sched_type=="ReduceOnPlateau":
                scheduler.step(avg_val)
                st.write(f"LR now: {optimizer.param_groups[0]['lr']:.2e}")

            if avg_val < best_val:
                best_val = avg_val
                best_state = model.state_dict()
                no_imp = 0
            else:
                no_imp += 1
                st.warning(f"No improvement for {no_imp} epoch(s).")
                if no_imp >= early_stop_pat:
                    st.warning("Early stopping.")
                    break

        # --------------------------------------------------------------
        # Restore best weights & show results
        # --------------------------------------------------------------
        model.load_state_dict(best_state)
        st.success("Training finished!")

        preds, trues = [], []
        model.eval()
        for batch in val_loader:
            ids = batch['input_ids'].to(device)
            msk = batch['attention_mask'].to(device)
            with torch.no_grad():
                out = model(ids, msk)
            preds.extend(np.argmax(out.cpu().numpy(),1))
            trues.extend(batch['labels'].numpy())

        f1 = f1_score(trues, preds, average="weighted")
        acc = accuracy_score(trues, preds)
        prec = precision_score(trues, preds, average="weighted", zero_division=0)
        rec = recall_score(trues, preds, average="weighted")
        st.write("Validation F1 Score:", f1)
        st.write("Accuracy:", acc)
        st.write("Precision:", prec)
        st.write("Recall:", rec)
        cm = confusion_matrix(trues, preds)
        st.subheader("Confusion Matrix")
        st.write(cm)

        # --------------------------------------------------------------
        # Save model & run logs
        # --------------------------------------------------------------
        os.makedirs("trained_models", exist_ok=True)
        run_idx = max(
            [int(d[:2]) for d in os.listdir("trained_models") if d[:2].isdigit()] or [0]
        ) + 1
        run_dir = f"trained_models/{run_idx:02d}_distilbert_epoch{epochs}"
        os.makedirs(run_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(run_dir, "model_checkpoint.pt"))
        os.makedirs("trained_models/currently_trained_dbert", exist_ok=True)
        torch.save(
            model.state_dict(),
            "trained_models/currently_trained_dbert/model_checkpoint.pt"
        )

        # Export HuggingFace-format fine-tuned model
        hf_export_dir = os.path.join("produced-models", "fine-tuned_dbert_politone")
        os.makedirs(hf_export_dir, exist_ok=True)
        export_config = DistilBertConfig.from_pretrained(base_path)
        export_config.num_labels = 3
        export_config.id2label = {0: "-1", 1: "0", 2: "1"}
        export_config.label2id = {"-1": 0, "0": 1, "1": 2}
        export_model = DistilBertForSequenceClassification(export_config)
        export_model.distilbert.load_state_dict(model.base_model.distilbert.state_dict())
        export_model.classifier.load_state_dict(model.classifier.state_dict())
        export_model.save_pretrained(hf_export_dir)
        tokenizer.save_pretrained(hf_export_dir)
        st.success(f"HuggingFace model saved to: {hf_export_dir}")

        # --------------------------------------------------------------
        # Save run info & update master log
        # --------------------------------------------------------------
        run_info = {
            "Run Index": run_idx,
            "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Epochs": epochs,
            "Batch Size": batch_size,
            **({"Learning Rate": learning_rate} if not enable_discriminative else {
                "Head LR": head_lr,
                "Top Layers LR": top_lr
            }),
            "Max Sequence Length": max_length,
            "Scheduler": sched_type,
            "Warmup Proportion": warmup_proportion,
            "OneCycle Max LR": max_lr_one,
            "OneCycle pct_start": pct_start,
            "Cosine T0": T0,
            "Cosine Tmult": Tmult,
            "ReduceOnPlateau Factor": reduce_fac,
            "ReduceOnPlateau Patience": reduce_pat,
            "Unfrozen Layers": num_unfreeze,
            "Final Training Loss": train_hist[-1],
            "Final Validation Loss": val_hist[-1],
            "Validation Accuracy": acc,
            "Validation F1": f1,
            "Validation Precision": prec,
            "Validation Recall": rec,
            "Enable Learnable Temp": enable_learnable_temp,
            "Learned Temperature": model.temp.item() if model.temp is not None else "",
            "Dropout Rate": dropout_rate,
            "Embedding Dropout": embedding_dropout,
            "Label Smoothing": label_smoothing,
            "Weight Decay": weight_decay,
            "Early Stopping Patience": early_stop_pat,
            "Enable Masking Augmentation": use_masking,
            "Masking Probability": mask_prob,
            "Enable Token Shuffling": use_shuffle,
            "Shuffle Window Size": shuffle_dist,
            "Temperature Scaling": temperature_scaling,
            "Windsorising Percentile": windsorise_threshold,
            "Repeat Threshold": repeat_thr,
            "Repeat Penalty Value": repeat_pen,
            "Repeat Maximum Penalty": repeat_max,
            "Loss Function": loss_fn,
            "Cooperative Weight": coop_w,
            "Neutral Weight": neutral_w,
            "Aggressive Weight": aggr_w,
            "Focal Gamma": focal_gamma if focal_gamma is not None else "",
            "Confidence Weighting": use_conf_weighting, 
        }
        pd.DataFrame([run_info]).to_excel(os.path.join(run_dir, "run_log.xlsx"), index=False)
        if os.path.exists(master_log_path):
            master_df = pd.read_excel(master_log_path)
            new_row = pd.DataFrame([run_info]).dropna(axis=1, how='all')
            master_df = pd.concat([master_df, new_row], ignore_index=True, sort=False)
        else:
            master_df = pd.DataFrame([run_info])
        master_df.sort_values("Run Index", inplace=True)
        master_df.to_excel(master_log_path, index=False)

        # --------------------------------------------------------------
        # Plots: loss curves & confusion matrix
        # --------------------------------------------------------------
        lc_path = os.path.join(run_dir, "loss_curve.svg")
        plt.figure()
        plt.plot(range(1,len(train_hist)+1), train_hist, label="Training")
        plt.plot(range(1,len(val_hist)+1), val_hist, label="Validation")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss Curves")
        plt.savefig(lc_path, format="svg")
        plt.close()

        cm_excel = os.path.join(run_dir, "confusion_matrix.xlsx")
        pd.DataFrame(cm, index=["Cooperative","Neutral","Aggressive"], columns=["Cooperative","Neutral","Aggressive"])\
        .to_excel(cm_excel)
        cm_path = os.path.join(run_dir, "confusion_matrix.svg")
        plt.figure(figsize=(6,6))
        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix"); plt.colorbar()
        ticks = np.arange(3)
        plt.xticks(ticks, ["Cooperative","Neutral","Aggressive"], rotation=45)
        plt.yticks(ticks, ["Cooperative","Neutral","Aggressive"])
        thresh = cm.max()/2.
        for i in range(3):
            for j in range(3):
                plt.text(j, i, cm[i,j], ha="center",
                        color="white" if cm[i,j]>thresh else "black")
        plt.savefig(cm_path, format="svg")
        plt.close()

        # --------------------------------------------------------------
        # Test data batch evaluation (post-processing/scoring options apply)
        # --------------------------------------------------------------
        test_dir = "testspeeches"
        test_rows = []
        if os.path.exists(test_dir):
            for fname in os.listdir(test_dir):
                if not fname.endswith(".txt"):
                    continue
                with open(os.path.join(test_dir, fname), "r", encoding="utf-8") as f:
                    speech_text = f.read()
                text = re.sub(r"\s+", " ", speech_text).strip()
                sentences = sent_tokenize(text)

                sentence_scores = []
                sentiment_scores = []

                for sentence in sentences:
                    inp = tokenizer.encode_plus(
                        sentence,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors="pt"
                    )
                    ids = inp['input_ids'].to(device)
                    msk = inp['attention_mask'].to(device)

                    with torch.no_grad():
                        logits = model(ids, msk)
                    probs = np.exp(logits.cpu().numpy().flatten() / temperature_scaling)
                    probs = probs / np.sum(probs)
                    score = compute_weighted_score(probs)
                    if use_conf_weighting:
                        score *= probs.max().item()
                    score = repeated_words_compensation(
                        sentence, score, repeat_thr, repeat_pen, repeat_max
                    )
                    sentence_scores.append(score)

                    with torch.no_grad():
                        sent_out = model(input_ids=ids, attention_mask=msk)
                    sent_probs = torch.softmax(sent_out, dim=-1).cpu().numpy().flatten()
                    sentiment_scores.append(sent_probs[1])

                sentence_scores = windsorise(np.array(sentence_scores), windsorise_threshold)

                overall_score = float(np.mean(sentence_scores)) if len(sentence_scores) else 0.0
                overall_sent = float(np.mean(sentiment_scores)) if len(sentiment_scores) else 0.0

                test_rows.append({
                    "Speech": fname,
                    "Score": overall_score,
                    "Sentiment Score": overall_sent
                })

            # Results dataframe and save
            test_df = pd.DataFrame(test_rows)
            test_df.to_excel(os.path.join(run_dir, "test_speeches_results.xlsx"), index=False)
            st.subheader("Test Speech Evaluation")
            st.dataframe(test_df)
        else:
            st.info("No test speeches found.")

        # --------------------------------------------------------------
        # Display plots and matrices
        # --------------------------------------------------------------
        st.subheader("Loss Curves")
        st.image(lc_path)
        st.subheader("Confusion Matrix")
        st.image(cm_path)
        st.success(f"Documentation and model checkpoint saved in folder: {run_dir}")
        
if __name__ == "__main__":
    run()
