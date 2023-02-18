### Exploration of SMC to dataset

We focus on analysing keyword frequency data which represent the number of daily online news articles that contain a specific
keyword. Multiple (up to 30–40) keywords are considered and the frequency of articles containing this keyword is analysed using Bayesian
methods.
We analyse inference methods for state-space models (or hidden Markov models) such as filtering and smoothing [1, 2]. This is done in order to study questions
about identifying trends (temporal correlations), correlations between
latent processes (correlations between keywords), and possibly identifying outliers or anomalies in the observed data. The data in question
represents counts that may be frequently zero in which case allowing
temporal correlations is crucial to be able to identify increasing or decreasing trend in data that is mostly zeros.