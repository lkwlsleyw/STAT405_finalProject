data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N, K] X;
  array[N] int<lower=0, upper=1> y;
}

parameters {
  real alpha;
  vector[K] beta;
}

model {
  vector[N] eta;
  
  alpha ~ normal(0, 2);
  beta ~ normal(0, 1.5);
  
  eta = alpha + X * beta;
  y ~ bernoulli_logit(eta);
}

generated quantities {
  array[N] int y_rep;
  vector[N] log_lik;
  
  for (n in 1:N) {
    real eta_n;
    eta_n = alpha + X[n] * beta;
    y_rep[n] = bernoulli_logit_rng(eta_n);
    log_lik[n] = bernoulli_logit_lpmf(y[n] | eta_n);
  }
}

