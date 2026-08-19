data {
  int<lower=1> K;
}
parameters {
  simplex[K] theta;
}
model {
  theta ~ dirichlet(rep_vector(1.0, K));
}
