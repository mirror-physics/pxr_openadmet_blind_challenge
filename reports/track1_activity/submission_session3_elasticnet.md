# Architecture

Ensemble of 17 ML models with an ElasticNetCV meta-learner from `scikit-learn`. The 17 base models include the 14 models from Session 2 plus 3 new models trained in Session 3:

**Session 2 models (14):**
- 9 Chemprop v2 D-MPNN models (3 training protocols &times; 3 random seeds each)
- 3 classical ML models&mdash;LightGBM, KRR, and RF&mdash;trained on RDKit Morgan fingerprints (radius=2, 2048 bits)
- 2 TorchMD-NET equivariant transformer (ET) models using CT-SCD pre-training (ChEMBL and single-concentration variants)

**Session 3 models (3 new):**
- 1 CT-SCD equivariant transformer variant with lr=3e-4 (vs 1e-4 default), SC pre-trained
- 2 ChemBERTa-77M-MLM SMILES transformer models (seeds 42, 123), fine-tuned with regression head

The ElasticNetCV meta-learner (`l1_ratio=0.50, alpha=0.0139`, selected via 5-fold CV) was trained on out-of-fold predictions from a 5-fold Butina cluster split (ECFP4, Tanimoto cutoff=0.4) of the sub-train set (3,730 compounds). The L1 regularization retained 10 models with non-zero weights:

| Model | Weight |
|-------|--------|
| chemprop_sc_seed2 | 0.320 |
| chemprop_sc_seed0 | 0.177 |
| ct_scd_chembl | 0.166 |
| chemprop_sc_seed1 | 0.148 |
| chemberta_seed123 | 0.076 |
| rf | 0.048 |
| chemberta_seed42 | 0.043 |
| chemprop_nr_seed0 | 0.038 |
| ct_scd_sc | 0.036 |
| vanilla_chemprop_seed2 | -0.035 |

# Training protocols

**Data splitting:** The full training set (4,139 compounds) was split 90/10 via Butina clustering into sub-train (3,730) and pseudo-test (409) sets. 5-fold CV used random splits of sub-train for Chemprop/classical ML and CT-SCD models.

**Chemprop and classical ML models** were trained as described in the Session 2 report (Optuna-optimized hyperparameters, three pre-training protocols).

**CT-SCD equivariant transformer models** used the TorchMD-NET architecture (`embedding_dim=256, num_layers=8, num_heads=8, cutoff=5.0 A`) with 3D conformers generated via RDKit ETKDG. The new lr=3e-4 variant used higher learning rate with cosine annealing and AdamW optimizer. Five additional CT-SCD variants (2 seed variations, 2 LR variations, 1 random init) were trained but received zero weight in the ensemble due to high correlation with existing CT-SCD models.

**ChemBERTa-77M-MLM** is a RoBERTa-based transformer (77M parameters) pre-trained on PubChem molecules via masked language modeling. Fine-tuned with a single regression output head on PXR pEC50 data. Training: `lr=2e-5, OneCycleLR schedule, batch_size=16, max 20 epochs, patience=5, fp16`. Two seeds (42, 123) were trained to provide seed diversity.

**Final predictions:** All base models were retrained on the full sub-train set (Chemprop + classical ML) or used as 5-fold ensembles (CT-SCD, ChemBERTa). The ElasticNetCV meta-learner applied learned weights to the 17-dimensional prediction vector on the 513 blinded test compounds.

# Key finding

Adding ChemBERTa (SMILES-sequence transformer) provided orthogonal signal to the ensemble, receiving ~11% total weight. In contrast, additional CT-SCD variants of the same architecture received zero weight due to correlation with existing models. This demonstrates that architectural diversity (graph MPNN + 3D equivariant + SMILES transformer) is more valuable than seed/hyperparameter diversity within a single architecture.
