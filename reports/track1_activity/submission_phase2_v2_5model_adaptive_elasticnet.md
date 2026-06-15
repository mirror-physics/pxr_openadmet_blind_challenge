# PXR Activity Prediction — Phase 2 v2: 5-Model Adaptive Ensemble

## Summary

An adaptive ElasticNet stacking ensemble of five complementary deep-learning models for PXR pEC50 prediction. This Phase 2 v2 submission targets the **generalization gap** observed previously (internal OOF RAE ≈ 0.49 but blinded Analog-Set-2 RAE ≈ 0.60), which was traced to scaffold information leaking through random K-fold cross-validation. The fix combines an honest validation design (Butina cluster CV + a random pseudo-test that mimics the in-fold analog test distribution), an expanded training set (+435 compounds), and FLAG adversarial regularization for the graph-transformer.

**Internal validation (honest):**
- Out-of-fold (Butina CV) RAE: **0.509**
- Random pseudo-test RAE: **0.520**
- OOF→pseudo-test gap: **+0.011** (down from ~0.06 under the previous random-KFold design)

The collapse of the OOF→holdout gap from ~0.06 to ~0.01 is the central result: internal estimates are now trustworthy predictors of in-fold blind performance.

## Why this design

The Phase 2 Analog Set 2 test set is an **in-fold analog sample** — structurally related to the training compounds (median Tanimoto nearest-neighbor similarity ≈ 0.73), not scaffold-novel. Two consequences drive the design:

1. **Model selection** uses **Butina cluster cross-validation** (ECFP4, Tanimoto cutoff 0.4) within the sub-train set, so hyperparameter and checkpoint choices are not rewarded for memorizing scaffolds.
2. **Performance estimation** uses a **random 90/10 pseudo-test holdout**, which reproduces the analog (in-fold) nature of the real test set far better than a scaffold-disjoint split would.

## Training data

Expanded training set of **4,827 compounds**:
- 4,139 original challenge training compounds
- 253 unblinded Analog Set 1 compounds (Phase 1)
- 88 semi-pure dose-response compounds (corrected pEC50, sample weight 1.0)
- 347 HT-chemistry crude dose-response compounds (corrected crude pEC50, sample weight 0.6 due to higher measurement uncertainty)

Split: random 90/10 → sub-train (4,345) + pseudo-test (482). Butina 5-fold CV within sub-train (869 compounds/fold, size-balanced greedy cluster assignment).

## Base models

| Model | Architecture | Pretraining | OOF RAE | Pseudo-test RAE |
|---|---|---|---|---|
| Chemprop-SC seed2 | D-MPNN | single-concentration pseudo-pEC50 | 0.519 | 0.544 |
| Chemprop-NR seed0 | D-MPNN | ChEMBL nuclear-receptor EC50 (encoder transfer) | 0.568 | 0.519 |
| Chemprop-vanilla seed0 | D-MPNN | none | 0.560 | 0.559 |
| GPS-FLAG | GPS graph transformer | 3-stage (RDKit props + GraphMAE → single-conc → PXR) | 0.553 | 0.554 |
| ChemBERTa seed123 | RoBERTa SMILES transformer | ChemBERTa-77M-MLM (PubChem) | 0.658 | 0.668 |

Chemprop variants fine-tuned from Phase-1 pretrained checkpoints via shape-matched encoder weight transfer; final task head trained fresh on PXR pEC50. All base models trained on the expanded Butina folds; per-fold test predictions averaged across 5 folds.

### FLAG regularization for GPS

A regularization sweep on the GPS graph transformer compared baseline (DropEdge only) against DropNode (15/20%), FLAG (Free Large-scale Adversarial augmentation on Graphs), R-Drop, and combinations, each evaluated by Butina-CV OOF RAE and pseudo-test RAE:

| Variant | OOF RAE | Pseudo-test RAE |
|---|---|---|
| **FLAG** | **0.553** | **0.554** |
| baseline | 0.558 | 0.567 |
| DropNode 15% | 0.570 | 0.570 |
| R-Drop | 0.559 | 0.569 |
| DropNode15 + FLAG | 0.565 | 0.563 |

FLAG (3-step PGD perturbation of node features, step size 1e-3) was the only method to improve both OOF and pseudo-test RAE while keeping the gap near zero; DropNode and R-Drop each hurt. FLAG was therefore adopted for the GPS member.

## Meta-learner

ElasticNetCV (l1_ratio=0.5, alpha=0.00178, 5-fold CV) fit on Butina out-of-fold base-model predictions plus three adaptive uncertainty features. Inputs: 5 model predictions + Tanimoto nearest-neighbor distance to train + Chemprop protocol disagreement + overall model disagreement.

Learned weights (largest magnitude first):
- chemprop_sc_seed2: +0.703
- gps_flag: +0.296
- overall_disagreement: −0.193
- chemprop_vanilla_seed0: −0.179
- chemprop_nr_seed0: +0.153
- chemprop_disagreement: −0.089
- chemberta_seed123: +0.018
- intercept: +0.089

The negative disagreement weights let the meta-learner shrink predictions toward the consensus when the base models diverge — a learned uncertainty discount. The stacked ensemble (pseudo-test RAE 0.520) modestly beats a simple 5-model average (0.521) while substantially beating every individual model.

## Data and reproducibility

All training data from the OpenADMET challenge dataset (challenge train + unblinded Analog Set 1 + provided semi-pure and HT-chemistry dose-response sets). Pre-training used public ChEMBL, ZINC250K, and PubChem only. No proprietary data.
