data { int<lower=1> K; }
parameters { ordered[K] x; }
model { x ~ normal(0, 2); }
