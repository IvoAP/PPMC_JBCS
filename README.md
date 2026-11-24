# PPMC_DATA_SUS
teste

## Membership inference

The loss-based attack assumes access to prediction probabilities and the true label.
It reports ROC AUC for distinguishing training members from held-out non-members;
0.5 corresponds to random guessing. Identical feature vectors are kept in the same
split. For each of five seeded splits, RPAR and the noise scale are fitted using only
the training data. The original-data reference uses the same split.

The attack evaluates the default classifier parameters, not the parameters tuned
elsewhere on the full dataset, to avoid exposing non-members during model selection.
Each `figures/mia_noise_<model>.png` shows mean AUC and standard deviation across
splits for each noise level, alongside the original-data baseline. These repeated
splits are not confidence intervals and the attack is not a general privacy guarantee.
