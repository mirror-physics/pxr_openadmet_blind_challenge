# Architecture

Ensemble of 17 ML models with a meta-learner taking their predictions and producing a final pEC50 estimate. Ensemble consists of:

- Three classical ML models&mdash;GBR, RF, and KRR&mdash;trained only on the provided training data all from `scikit-learn`
- 14 Chemprop models trained on different data (details below) related to the pEC50 data in PXR from the challenge

The meta-learner was a `RidgeCV` learner from `scikit-learn` that took predictions from the model stack, weighed their inputs, and produced an improved guess at the pEC50 score.

# Training protocols

The classical ML models were trained only on the pEC50 training data provided by OpenADMET. The Chemprop models were trained in three different ways, where model initialized with different random seeds were used to form ensembles from each training category:

- 5 Chemprop models were trained from scratch only on the pEC50 data
- 5 Chemprop models were trained on a PXR pEC50 data from ChEMBL in addition to the pEC50 data from OpenADMET
- 4 Chemprop models were trained only using the ~21k single-concentration (SC) data points, which were fitted with sigmoid regression to estimate pEC50 for each of the compounds
