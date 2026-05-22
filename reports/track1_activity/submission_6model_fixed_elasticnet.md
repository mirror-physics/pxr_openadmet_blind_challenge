# Architecture

Fixed-weight ensemble of 6 ML models with an ElasticNetCV meta-learner from `scikit-learn`. The meta-learner receives only the 6 model predictions as input (no compound-specific features).

## Base models

Same 6 models as the adaptive variant (see `submission_6model_adaptive_elasticnet.md` for full details):

1. **Chemprop-SC seed 2** -- D-MPNN, single-concentration pre-trained (out-of-fold RAE: 0.517)
2. **Chemprop-NR seed 0** -- D-MPNN, ChEMBL NR pre-trained (OOF RAE: 0.549)
3. **Chemprop-vanilla seed 0** -- D-MPNN, no pre-training (OOF RAE: 0.544)
4. **CT-SCD-ChEMBL** -- TorchMD-NET equivariant transformer, PCQ + ChEMBL pre-trained (OOF RAE: 0.591)
5. **GPS pre-trained regularized** -- GPS graph transformer, 3-stage pre-training on 3M molecules with GraphMAE + single-conc + PXR (OOF RAE: 0.534)
6. **ChemBERTa seed 123** -- RoBERTa SMILES transformer, 77M PubChem MLM pre-trained (OOF RAE: 0.614)

The 6 models span 4 distinct architectures (D-MPNN, 3D equivariant transformer, 2D graph transformer, SMILES sequence transformer) and 3 Chemprop pre-training protocols to maximize representation diversity.

## Meta-learner

ElasticNetCV (selected via 5-fold CV). Learned weights:
- chemprop_sc_seed2: +0.451 (45%)
- gps_pretrained_reg: +0.291 (29%)
- ct_scd_chembl: +0.178 (18%)
- chemberta_seed123: +0.064 (6%)
- chemprop_nr_seed0: +0.051 (5%)
- vanilla_chemprop_seed0: -0.034 (-3%)

# Training protocol

**Data splitting:** Full training set (4,139 compounds) split 90/10 via Butina clustering (ECFP4, Tanimoto cutoff=0.4) into sub-train (3,730) and pseudo-test (409). 5-fold random CV on sub-train. Meta-learner trained on out-of-fold predictions.

**No proprietary data used.** Pre-training used public ChEMBL, ZINC250K, PubChem, and PCQM4MV2 data.

# Internal validation

Out-of-fold (OOF) RAE: 0.4826. OOF predictions are generated via 5-fold cross-validation on the sub-train set, where each compound's prediction comes from a model that did not see it during training. RAE (Relative Absolute Error) is MAE divided by the MAE of a mean-prediction baseline; values below 1.0 indicate the model outperforms the baseline.
