data { int<lower=1> K; }
parameters { cholesky_factor_cov[K] L; }
model { to_vector(L) ~ normal(0, 2); }
