data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> J;
  matrix[N, K] X;
  array[N] int<lower=1, upper=J> month;
  array[N] int<lower=0, upper=1> y;
}

parameters {
  real mu_alpha;
  real<lower=0> sigma_alpha;
  vector[J] alpha_raw;
  vector[K] beta;
}

transformed parameters {
  vector[J] alpha;
  alpha = mu_alpha + sigma_alpha * alpha_raw;
}

model {
  vector[N] eta;
  
  mu_alpha ~ normal(0, 2);
  sigma_alpha ~ exponential(1);
  alpha_raw ~ normal(0, 1);
  beta ~ normal(0, 1.5);
  
  eta = alpha[month] + X * beta;
  y ~ bernoulli_logit(eta);
}

generated quantities {
  array[N] int y_rep;
  vector[N] log_lik;
  
  for (n in 1:N) {
    real eta_n;
    eta_n = alpha[month[n]] + X[n] * beta;
    y_rep[n] = bernoulli_logit_rng(eta_n);
    log_lik[n] = bernoulli_logit_lpmf(y[n] | eta_n);
  }
}

