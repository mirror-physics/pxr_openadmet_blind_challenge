# Architecture

TorchMD-NET equivariant transformer (ET) model for 3D molecular property prediction, using self-conditioned denoising (CT-SCD) pre-training from Perez et al. This model operates on 3D atomic coordinates rather than 2D molecular graphs, providing a complementary signal to MPNN-based approaches.

Architecture: `embedding_dim=256, num_layers=8, num_heads=8, num_rbf=64, cutoff=5.0 A, activation=SiLU, aggregation=add, O(3) equivariance`.

Test predictions are an ensemble average of 5 fold checkpoint models from OOF cross-validation.

# Training protocols

**Data splitting:** The full training set (4,139 compounds) was split 90/10 via Butina clustering (ECFP4, Tanimoto cutoff=0.4) into sub-train (3,730) and pseudo-test (409) sets. 5-fold CV used random splits of sub-train.

**3D conformer generation:** RDKit ETKDG was used to generate a single 3D conformer per compound. 1 of 3,730 sub-train compounds and 1 of 409 pseudo-test compounds failed conformer generation; these received mean-imputed predictions.

**Pre-training chain:** PCQ (self-conditioned denoising) &rarr; single-concentration PXR sigmoid-fitted pEC50 data. The SC pre-training step fine-tuned the PCQ-denoised weights on ~21k single-concentration PXR data points with sigmoid-estimated pEC50 values.

**Fine-tuning on pEC50:** Each of 5 CV folds was fine-tuned with `lr=1e-4, AdamW (weight_decay=1e-5), cosine annealing (eta_min=1e-6), batch_size=32, patience=20, max_epochs=100, grad_clip=1.0, MSE loss`. Average OOF MAE across folds: 0.5502.

**Test predictions:** Rather than retraining a single model on the full sub-train set, predictions from all 5 fold checkpoints were averaged on each test compound. This fold-ensemble approach provides built-in regularization.

**Software:** `torchmd-net`, `torch_geometric`, `rdkit`

