# Epileptic Seizure Onset Prediction Using EEG Data

In this project, I've designed and implemented a mathematical model for the prediction of seizure onset using EEG data based on the notion of Functional Brain Connectivity as measured by the Granger Causality index.

In the preprocessing step, standard filtering and frequency-band extraction has been performed and the data has been transformed into the Common Average Reference framework. Afterwards, the Independent Component Analysis method has been applied to the data and the corresponding components have been extracted. The Granger causality measure in the frequency domain is then calculated for a sliding window and used as a predictor variable for an L1-regularized Logistic Regression classifier. The t-SNE method is used for the visualization of the high-dimensional connectivity features.

Please refer to the following paper for a recent review of the relevant literature:
Kuhlmann, Levin, et al. "Seizure prediction—ready for a new era." Nature Reviews Neurology 14.10 (2018): 618-630.
