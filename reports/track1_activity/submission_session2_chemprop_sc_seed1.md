# Architecture

Single Chemprop v2 D-MPNN (directed message-passing neural network) model for pEC50 regression. This is the best-performing individual model from the 14-model ensemble, selected based on pseudo-test evaluation.

# Training protocols

**Data splitting:** The full training set (4,139 compounds) was split 90/10 via Butina clustering (ECFP4, Tanimoto cutoff=0.4) into sub-train (3,730) and pseudo-test (409) sets for model selection. The final model was retrained on the full sub-train set.

**Pre-training:** The model was initialized from a checkpoint pre-trained on ~21k single-concentration (SC) PXR data points. SC data was fitted with sigmoid regression to estimate pEC50 for each compound, providing a transfer learning starting point.

**Fine-tuning on pEC50:** Optuna-optimized hyperparameters were used: `depth=5, message_hidden_dim=250, ffn_hidden_dim=300, ffn_num_layers=2, dropout=0.3, batch_size=32, max_lr=1.67e-3, init_lr=2.71e-4, final_lr=4.40e-5, warmup_epochs=4, patience=15, max_epochs=100`. This was seed 1 of 4 SC-initialized models; seed selection was based on OOF performance during 5-fold CV.

**Software:** `chemprop>=2.1.0`

