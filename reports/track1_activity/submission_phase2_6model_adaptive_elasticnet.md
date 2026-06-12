# Architecture

Adaptive ensemble of 6 ML models with an ElasticNetCV meta-learner from `scikit-learn`. The meta-learner receives 9 input features per compound: 6 model predictions plus 3 compound-specific uncertainty features (Tanimoto nearest-neighbor distance, Chemprop protocol disagreement, and overall model disagreement). This allows the linear meta-learner to adjust its effective weighting based on how novel or uncertain each compound is.

## Base models

**1. Chemprop-SC seed 2** (D-MPNN, single-concentration pre-trained)
- Directed message-passing neural network (Chemprop v2)
- Pre-trained on 21K single-concentration pseudo-pEC50 values, fine-tuned on PXR dose-response pEC50
- Optuna-optimized hyperparameters: depth=5, message_hidden_dim=250, ffn_hidden_dim=300, ffn_num_layers=2, dropout=0.3
- Out-of-fold (OOF) RAE: 0.514

**2. Chemprop-NR seed 0** (D-MPNN, ChEMBL nuclear receptor pre-trained)
- Same architecture as above
- Pre-trained on ChEMBL nuclear receptor EC50 data across PXR, FXR, LXRa, LXRb
- OOF RAE: 0.542

**3. Chemprop-vanilla seed 0** (D-MPNN, no pre-training)
- Trained from scratch on PXR dose-response pEC50 only
- OOF RAE: 0.544

**4. CT-SCD-ChEMBL** (TorchMD-NET equivariant transformer)
- Architecture: embedding_dim=256, num_layers=8, num_heads=8, cutoff=5.0 A, O(3) equivariance
- Pre-trained via self-conditioned denoising on PCQM4MV2 (3.4M molecules, Perez et al.), then fine-tuned on ChEMBL PXR data, then on PXR dose-response pEC50
- 3D conformers generated via RDKit ETKDG
- OOF RAE: 0.596

**5. GPS pre-trained regularized** (GPS graph transformer)
- Architecture: 4 GPSConv layers (GATv2 local MPNN + global multi-head attention), hidden_dim=128, 4 heads
- 3-stage pre-training:
  - Stage 0: 20 epochs on 3.08M molecules (ChEMBL + ZINC250K) with dual objectives -- multi-task regression on 10 RDKit properties (MolWt, LogP, TPSA, HBA, HBD, RotBonds, HeavyAtoms, AromaticRings, FractionCSP3, QED) + GraphMAE attribute masking (mask 15% of atom types, predict via decoder, scaled cross-entropy loss)
  - Stage 1: 20 epochs fine-tuning on 21K single-concentration pseudo-pEC50
  - Stage 2: 10 epochs fine-tuning on PXR dose-response pEC50 with regularization (discriminative LR: 1e-4 backbone / 1e-3 head, weight_decay=1e-3, DropEdge 10%)
- OOF RAE: 0.539

**6. ChemBERTa seed 123** (RoBERTa SMILES transformer)
- ChemBERTa-77M-MLM: 77M parameter RoBERTa pre-trained on PubChem via masked language modeling
- Fine-tuned with single regression head on PXR pEC50
- Discriminative LR (2e-5 backbone, 1e-3 head), cosine schedule with 10% warmup, weight_decay=1e-2, batch_size=32, max 15 epochs, patience=5
- OOF RAE: 0.638

## Adaptive uncertainty features

**Tanimoto NN distance:** For each compound, the maximum Tanimoto similarity (Morgan fingerprints, radius=2, 2048 bits) to any compound in the training set. Enables the meta-learner to down-weight predictions for novel chemistry (low similarity to training data).

**Chemprop protocol disagreement:** Standard deviation across the 3 Chemprop variants (SC-pretrained, NR-pretrained, vanilla). High disagreement indicates the compound lies in a region where pre-training data matters, signaling higher uncertainty.

**Overall model disagreement:** Standard deviation across all 6 model predictions. High disagreement indicates the compound is challenging for the ensemble.

## Meta-learner

ElasticNetCV (`l1_ratio=0.10, alpha=0.00518`, selected via 5-fold CV). Learned weights:
- chemprop_sc_seed2: +0.515
- gps_pretrained_reg: +0.334
- ct_scd_chembl: +0.168
- chemprop_nr_seed0: +0.159
- tanimoto_nn_dist: +0.089
- chemberta_seed123: +0.025
- chemprop_disagreement: -0.168 (penalize high protocol disagreement)
- overall_disagreement: -0.119 (penalize high model disagreement)
- chemprop_vanilla_seed0: -0.182
- intercept: -0.060

# Training protocol

**Phase 2 expanded dataset:** Full training set of 4,392 compounds, combining the original 4,139-compound training set with 253 unblinded Analog Set 1 compounds from Phase 1. Split 90/10 via Butina clustering (ECFP4, Tanimoto cutoff=0.4) into sub-train (3,952) and pseudo-test (440). 5-fold random CV on sub-train for all models. The meta-learner was trained on out-of-fold predictions to prevent information leakage.

**No proprietary data used.** All training data from the OpenADMET challenge dataset. Pre-training used public ChEMBL, ZINC250K, and PubChem data.

# Internal validation

Out-of-fold (OOF) RAE: 0.4913. OOF predictions are generated via 5-fold cross-validation on the sub-train set, where each compound's prediction comes from a model that did not see it during training. RAE (Relative Absolute Error) is MAE divided by the MAE of a mean-prediction baseline; values below 1.0 indicate the model outperforms the baseline.
