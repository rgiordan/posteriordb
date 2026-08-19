data {
  real mu;
}
parameters {
  real y;
}
model {
  y ~ normal(mu, 1);
}
