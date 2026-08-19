data { int<lower=1> K; }
parameters { cov_matrix[K] S; }
model { S ~ inv_wishart(K + 1.0, diag_matrix(rep_vector(1.0, K))); }
