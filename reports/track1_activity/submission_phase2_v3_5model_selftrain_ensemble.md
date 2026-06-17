# PXR Activity Prediction — Phase 2 v3: 5-Model Ensemble with Self-Trained GPS

## Summary

An adaptive ElasticNet stacking ensemble of five complementary deep-learning models for PXR pEC50 prediction. This Phase 2 v3 submission builds on v2 (Butina cluster CV, expanded training data, FLAG regularization) by replacing the GPS-FLAG base model with a **self-trained GPS** variant that uses pseudo-labeled single-concentration compounds to augment training. Three additional Phase 2 experiments (GraphMAE+contrastive pre-training, multi-task learning, self-training) were evaluated; only self-training improved generalization.

**Internal validation:**
- Out-of-fold (Butina CV) RAE: **0.508**
- Random pseudo-test RAE: **0.517**
- OOF→pseudo-test gap: **+0.009**

**Improvement over v2:** pseudo-test RAE decreased from 0.520 → 0.517 (delta −0.003), and the OOF→holdout gap narrowed from +0.011 to +0.009.

## Phase 2 experiments

Four advanced training methods were evaluated on top of the Phase 1 baseline (Butina CV + FLAG + expanded data). Each was tested using Butina CV OOF RAE and random pseudo-test RAE:

| Experiment | OOF RAE | PT RAE | Delta PT vs baseline | Verdict |
|---|---|---|---|---|
| **2B. GraphMAE + contrastive pre-training** | 0.566 | 0.561 | +0.007 | Negative |
| **2C. Multi-task (pEC50 + conc-response)** | 0.553 | 0.554 | −0.000 | Neutral |
| **2D. Self-training with pseudo-labels** | 0.550 | 0.548 | −0.006 | Positive |

Baseline (GPS-FLAG): OOF 0.553, PT 0.554.

### 2B. GraphMAE + contrastive domain adaptation (negative)
Joint self-supervised pre-training combining GraphMAE (masked atom-type reconstruction) with InfoNCE contrastive loss (subgraph dropping + feature masking augmentations). Trained on ~15K molecules (expanded train + novel single-conc + 2,000 domain-bridging ZINC250K compounds selected by Tanimoto proximity to test). The adapted representations *worsened* downstream pEC50 performance (+0.007 PT RAE), possibly because contrastive pre-training with small-graph augmentations disrupted the functional group-level features most relevant to PXR binding.

### 2C. Multi-task learning with single-concentration data (neutral)
Auxiliary concentration-response head (`[embedding, log10(conc)] → MLP → log2FC`) alongside the primary pEC50 head, with FLAG applied to pEC50 batches only and λ_conc = 0.15. The auxiliary signal from ~21K single-concentration measurements neither helped nor hurt the primary task (delta −0.0004 PT RAE).

### 2D. Self-training with pseudo-labels (positive)
Used the Phase 1 GPS-FLAG 5-fold ensemble to pseudo-label 8,134 novel single-concentration compounds (ensemble std < 0.35 filter, 99.99% passed). Added to training with sample weight 0.2, then retrained GPS with FLAG on the combined data. This was the only Phase 2 method to improve generalization (−0.006 PT RAE), likely because the pseudo-labeled compounds expand chemical coverage in regions where the ensemble is already confident, providing a soft form of domain adaptation.

## Training data

Expanded training set of **4,827 compounds** (unchanged from v2):
- 4,139 original challenge training compounds
- 253 unblinded Analog Set 1 compounds (Phase 1)
- 88 semi-pure dose-response compounds (corrected pEC50, sample weight 1.0)
- 347 HT-chemistry crude dose-response compounds (corrected crude pEC50, sample weight 0.6)

Additionally, 8,134 novel single-concentration compounds with GPS-FLAG ensemble pseudo-labels were used for GPS self-training only (sample weight 0.2).

Split: random 90/10 → sub-train (4,345) + pseudo-test (482). Butina 5-fold CV within sub-train (ECFP4 2048-bit, Tanimoto cutoff 0.4, greedy size-balanced cluster assignment).

## Base models

| Model | Architecture | Pretraining | OOF RAE | Pseudo-test RAE |
|---|---|---|---|---|
| Chemprop-SC seed2 | D-MPNN | single-concentration pseudo-pEC50 | 0.519 | 0.544 |
| Chemprop-NR seed0 | D-MPNN | ChEMBL nuclear-receptor EC50 (encoder transfer) | 0.568 | 0.519 |
| Chemprop-vanilla seed0 | D-MPNN | none | 0.560 | 0.559 |
| **GPS-selftrain** | GPS graph transformer | 3-stage + self-training (pseudo-labels, weight 0.2) | **0.550** | **0.548** |
| ChemBERTa seed123 | RoBERTa SMILES transformer | ChemBERTa-77M-MLM (PubChem) | 0.658 | 0.668 |

GPS-selftrain uses 3-stage pretraining (RDKit props + GraphMAE → single-conc → PXR pEC50 with FLAG), then retrains on the combined labeled + pseudo-labeled data with FLAG adversarial perturbation (3-step PGD, step_size=1e-3, discriminative LR: 1e-4 backbone, 1e-3 head).

## Meta-learner

ElasticNetCV (l1_ratio=0.3, alpha=0.00297, 5-fold CV) fit on Butina out-of-fold base-model predictions plus three adaptive uncertainty features. Inputs: 5 model predictions + Tanimoto nearest-neighbor distance to train + Chemprop protocol disagreement + overall model disagreement.

Learned weights (largest magnitude first):
- chemprop_sc_seed2: +0.685
- gps_selftrain: +0.321
- overall_disagreement: −0.162
- chemprop_vanilla_seed0: −0.172
- chemprop_nr_seed0: +0.152
- chemprop_disagreement: −0.092
- chemberta_seed123: +0.011
- tanimoto_nn_dist: +0.000
- intercept: +0.057

The ensemble (pseudo-test RAE 0.517) beats every individual model and the simple 5-model average (0.521).

## Data and reproducibility

All training data from the OpenADMET challenge dataset (challenge train + unblinded Analog Set 1 + provided semi-pure and HT-chemistry dose-response sets). Pre-training used public ChEMBL, ZINC250K, and PubChem only. No proprietary data. Self-training pseudo-labels derived from the provided single-concentration screening data using the GPS-FLAG ensemble trained on the challenge data.
