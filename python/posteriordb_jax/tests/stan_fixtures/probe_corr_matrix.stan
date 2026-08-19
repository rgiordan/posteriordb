data { int<lower=1> K; }
parameters { corr_matrix[K] Omega; }
model { Omega ~ lkj_corr(1.0); }
