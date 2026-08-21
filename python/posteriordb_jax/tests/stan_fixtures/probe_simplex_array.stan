data {
  int<lower=1> M;
  int<lower=1> K;
}
parameters {
  array[M] simplex[K] theta;
}
model {
  for (m in 1 : M) {
    theta[m] ~ dirichlet(rep_vector(1.0, K));
  }
}
