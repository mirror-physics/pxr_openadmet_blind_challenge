# Architecture

Ensemble of 20 ML models with an ElasticNetCV meta-learner from `scikit-learn`. The models span 4 distinct architectures:

**Chemprop D-MPNN (9 models, ~63% total weight):**
- 9 Chemprop v2 directed message-passing neural network models across 3 training protocols (vanilla, ChEMBL NR pre-trained, single-concentration pre-trained) &times; 3 seeds each
- Optuna-optimized hyperparameters: depth=5, message_hidden_dim=250, ffn_hidden_dim=300

**TorchMD-NET Equivariant Transformer (3 models, ~20% total weight):**
- CT-SCD-ChEMBL: pre-trained via self-conditioned denoising on PCQM4MV2 (3.4M molecules), fine-tuned on ChEMBL PXR data
- CT-SCD-SC: pre-trained on PCQM4MV2, fine-tuned on single-concentration PXR data
- CT-SCD-SC-lr3e4: same as CT-SCD-SC but with learning rate 3e-4 (vs default 1e-4)
- Architecture: embedding_dim=256, num_layers=8, num_heads=8, cutoff=5.0 A, O(3) equivariance

**GPS Graph Transformer (2 models, ~19% total weight):**
- General, Powerful, Scalable (GPS) graph transformer combining GATv2 local message-passing with global multi-head self-attention
- Architecture: 4 GPSConv layers, hidden_dim=128, 4 attention heads, GATv2 local MPNN
- Atom features: one-hot encoded atomic number, degree, formal charge, Hs, hybridization, aromaticity, ring membership (146-dim)
- 2 seeds (0, 1) for ensemble diversity
- OOF RAE: 0.5764 (seed 0), 0.5658 (seed 1)

**ChemBERTa SMILES Transformer (3 models, ~10% total weight):**
- ChemBERTa-77M-MLM: RoBERTa-based transformer (77M parameters) pre-trained on PubChem via masked language modeling
- 2 plain fine-tuned models (seeds 42, 123): OOF RAE ~0.6134
- 1 foundation-pretrained model: additional multi-task pre-training on 10 computed molecular properties (MolWt, LogP, TPSA, HBA, HBD, RotBonds, HeavyAtoms, AromaticRings, FractionCSP3, QED) from 12,616 molecules, then fine-tuned on PXR. OOF RAE 0.5953 (1.8% improvement over plain ChemBERTa)

**Classical ML (1 model, ~2% weight):**
- Random Forest on ECFP4 Morgan fingerprints (radius=2, 2048 bits)

The ElasticNetCV meta-learner (`l1_ratio=0.50`) retains 16 of 20 models with non-zero weights. Top contributors: chemprop_sc_seed2 (29.6%), chemprop_sc_seed0 (17.0%), ct_scd_chembl (16.5%), chemprop_sc_seed1 (14.2%), gps_seed1 (13.5%).

# Training protocols

**Data splitting:** Full training set (4,139 compounds) split 90/10 via Butina clustering (ECFP4, Tanimoto cutoff=0.4) into sub-train (3,730) and pseudo-test (409). 5-fold CV on sub-train for all models.

**GPS training:** AdamW optimizer, lr=1e-3, CosineAnnealingLR, batch_size=64, max 80 epochs, patience=15, gradient clipping=1.0. Target normalization: per-fold z-scoring.

**ChemBERTa training:** AdamW optimizer, lr=2e-5, OneCycleLR with 10% warmup, batch_size=16, max 20 epochs, patience=5, fp16 mixed precision.

**Foundation pre-training:** 10 epochs of multi-task regression on 12,616 molecules with 10 RDKit-computed properties (z-score normalized). Starting from ChemBERTa-77M-MLM weights, lr=5e-5, CosineAnnealingLR.

**Final predictions:** All base models use 5-fold checkpoint ensembling (average predictions from 5 fold-specific models). The ElasticNetCV meta-learner combines the 20-dimensional prediction vector.

# Key findings

1. **Architectural diversity is the primary driver of ensemble improvement.** GPS (graph transformer) and ChemBERTa (SMILES transformer) each received substantial ensemble weight (~19% and ~10% respectively), while additional CT-SCD variants of the same architecture received near-zero weight.

2. **GPS is the strongest new architecture.** GPS seed 1 achieved OOF RAE 0.5658 and received 13.5% ensemble weight -- the highest of any new model.

3. **Foundation pre-training provides marginal improvement.** Multi-task property pre-training improved ChemBERTa from RAE 0.6134 to 0.5953 (+1.8%), worth ~1.7% ensemble weight.

# OOF RAE: 0.4781
