data {
  int<lower=1> K;
}
parameters {
  cholesky_factor_corr[K] L;
}
model {
  L ~ lkj_corr_cholesky(1.0);
}
