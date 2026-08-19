data { int<lower=1> dummy; }
parameters {
  real<lower=1.5> lo;
  real<upper=-2.0> hi;
  real<lower=-1.0, upper=3.0> both;
}
model {
  lo ~ normal(0, 3);
  hi ~ normal(0, 3);
  both ~ normal(0, 3);
}
