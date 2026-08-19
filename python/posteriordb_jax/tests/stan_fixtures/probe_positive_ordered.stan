data { int<lower=1> K; }
parameters { positive_ordered[K] x; }
model { x ~ normal(0, 2); }
