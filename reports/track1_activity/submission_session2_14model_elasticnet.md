# Architecture

Ensemble of 14 ML models with an ElasticNetCV meta-learner from `scikit-learn` taking their out-of-fold predictions and producing a final pEC50 estimate. The 14 base models are:

- 9 Chemprop v2 D-MPNN models (3 training protocols &times; 3 random seeds each)
- 3 classical ML models&mdash;LightGBM, KRR, and RF&mdash;trained on RDKit Morgan fingerprints (radius=2, 2048 bits)
- 2 TorchMD-NET equivariant transformer (ET) models using CT-SCD pre-training from Perez et al., initialized from different pre-trained checkpoints (ChEMBL and single-concentration)

The ElasticNetCV meta-learner (`l1_ratio=0.7, alpha=0.0087`, selected via 5-fold CV) was trained on out-of-fold predictions from a 5-fold Butina cluster split (ECFP4, Tanimoto cutoff=0.4) of the sub-train set (3,730 compounds). The L1 regularization zeroed out several weak models, retaining primarily: Chemprop-SC seeds (total weight ~0.66), CT-SCD-ChEMBL (0.18), RF (0.08), and Chemprop-NR seed0 (0.05).

# Training protocols

**Data splitting:** The full training set (4,139 compounds) was split 90/10 via Butina clustering into sub-train (3,730) and pseudo-test (409) sets. 5-fold CV used random splits of sub-train.

**Chemprop models** were trained with Optuna-optimized hyperparameters: `depth=5, message_hidden_dim=250, ffn_hidden_dim=300, ffn_num_layers=2, dropout=0.3, batch_size=32, max_lr=1.67e-3, warmup_epochs=4, patience=15`. Three training protocols:

- 5 vanilla models trained from scratch on pEC50 data only (3 seeds selected)
- 5 models initialized from a ChEMBL PXR NR-assay pre-trained checkpoint (3 seeds)
- 4 models initialized from a single-concentration PXR sigmoid-fitted pEC50 checkpoint (3 seeds)

**CT-SCD equivariant transformer models** used the TorchMD-NET architecture (`embedding_dim=256, num_layers=8, num_heads=8, cutoff=5.0 A`) with 3D conformers generated via RDKit ETKDG. Two variants with different pre-training:

- CT-SCD-ChEMBL: pre-trained on PCQ via self-conditioned denoising, then fine-tuned on ChEMBL PXR data
- CT-SCD-SC: pre-trained on PCQ, then fine-tuned on single-concentration PXR data

Both were fine-tuned on sub-train pEC50 with `lr=1e-4, AdamW, cosine annealing, patience=20`. For test predictions, 5 fold checkpoints were ensembled rather than retraining on full data.

**Classical ML models** used scikit-learn defaults with Morgan fingerprint features, trained only on the provided pEC50 data.

**Final predictions:** All 14 base models were retrained on the full sub-train set (Chemprop + classical ML) or used as 5-fold ensembles (CT-SCD). The ElasticNetCV meta-learner applied learned weights to the 14-dimensional prediction vector on the 513 blinded test compounds.
