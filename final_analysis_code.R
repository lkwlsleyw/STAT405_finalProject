library(cmdstanr)
library(posterior)
library(MASS)

set.seed(123)

# data
df <- read.csv("online_shoppers_intention.csv")

df_final <- df[, c(
  "Revenue",
  "Month",
  "PageValues",
  "BounceRates",
  "ExitRates",
  "ProductRelated_Duration",
  "VisitorType",
  "Weekend"
)]

head(df_final)
summary(df_final)

colSums(is.na(df_final))
table(df_final$Revenue)
table(df_final$Month)
table(df_final$VisitorType)
table(df_final$Weekend)

# cleaning
df_final$Revenue <- as.integer(df_final$Revenue)
df_final$Weekend <- as.integer(df_final$Weekend)
df_final$ReturningVisitor <- ifelse(df_final$VisitorType == "Returning_Visitor", 1, 0)
df_final$MonthIndex <- as.integer(factor(df_final$Month))

month_key <- data.frame(
  Month = levels(factor(df_final$Month)),
  MonthIndex = 1:length(levels(factor(df_final$Month)))
)

str(df_final)
table(df_final$Revenue)
table(df_final$Weekend)
table(df_final$ReturningVisitor)
table(df_final$MonthIndex)
month_key

# standardize
df_final$PageValues_std <- as.numeric(scale(df_final$PageValues))
df_final$BounceRates_std <- as.numeric(scale(df_final$BounceRates))
df_final$ExitRates_std <- as.numeric(scale(df_final$ExitRates))
df_final$ProductRelated_Duration_std <- as.numeric(scale(df_final$ProductRelated_Duration))

sapply(df_final[, c(
  "PageValues_std",
  "BounceRates_std",
  "ExitRates_std",
  "ProductRelated_Duration_std")], mean)

sapply(df_final[, c(
  "PageValues_std",
  "BounceRates_std",
  "ExitRates_std",
  "ProductRelated_Duration_std")], sd)

# simple summaries
mean(df_final$Revenue)

purchase_by_month <- aggregate(Revenue ~ Month, data = df_final, mean)
purchase_by_month <- merge(purchase_by_month, month_key, by = "Month")
purchase_by_month <- purchase_by_month[order(purchase_by_month$MonthIndex), ]
purchase_by_month

aggregate(Revenue ~ ReturningVisitor, data = df_final, mean)

aggregate(cbind(
  PageValues_std,
  BounceRates_std,
  ExitRates_std,
  ProductRelated_Duration_std
) ~ Revenue, data = df_final, mean)

plot(
  purchase_by_month$MonthIndex,
  purchase_by_month$Revenue,
  type = "b",
  xaxt = "n",
  xlab = "Month",
  ylab = "Purchase rate"
)
axis(1, at = purchase_by_month$MonthIndex, labels = purchase_by_month$Month)

# data for models
df_fit <- df_final[, c(
  "Revenue",
  "MonthIndex",
  "PageValues_std",
  "BounceRates_std",
  "ExitRates_std",
  "ProductRelated_Duration_std",
  "ReturningVisitor",
  "Weekend"
)]

X <- as.matrix(df_fit[, c(
  "PageValues_std",
  "BounceRates_std",
  "ExitRates_std",
  "ProductRelated_Duration_std",
  "ReturningVisitor",
  "Weekend"
)])

y <- df_fit$Revenue
month <- df_fit$MonthIndex

N <- nrow(X)
K <- ncol(X)
J <- length(unique(month))

coef_names <- c(
  "alpha",
  "PageValues_std",
  "BounceRates_std",
  "ExitRates_std",
  "ProductRelated_Duration_std",
  "ReturningVisitor",
  "Weekend"
)

# prior predictive
n_prior <- 1000
a_prior <- rnorm(n_prior, 0, 2)
b_prior <- matrix(rnorm(n_prior * K, 0, 1.5), nrow = n_prior, ncol = K)

prior_rate <- rep(NA, n_prior)

for (i in 1:n_prior) {
  eta <- a_prior[i] + X %*% b_prior[i, ]
  p <- plogis(eta)
  prior_rate[i] <- mean(p)
}

summary(prior_rate)

hist(
  prior_rate,
  breaks = 30,
  main = "prior predictive mean purchase rate",
  xlab = "mean purchase rate"
)
abline(v = mean(y), col = "red", lwd = 2)

# model 1 laplace
log_post_m1 <- function(par, X, y) {
  a <- par[1]
  b <- par[-1]
  eta <- as.vector(a + X %*% b)
  log_lik <- sum(y * eta - log1p(exp(eta)))
  log_prior <- dnorm(a, 0, 2, log = TRUE) + sum(dnorm(b, 0, 1.5, log = TRUE))
  log_lik + log_prior
}

neg_log_post_m1 <- function(par, X, y) {
  -log_post_m1(par, X, y)
}

laplace_fit <- optim(
  par = rep(0, K + 1),
  fn = neg_log_post_m1,
  X = X,
  y = y,
  method = "BFGS",
  hessian = TRUE
)

laplace_mode <- laplace_fit$par
laplace_cov <- solve(laplace_fit$hessian)
laplace_cov <- (laplace_cov + t(laplace_cov)) / 2
laplace_sd <- sqrt(diag(laplace_cov))

m1_laplace_summary <- data.frame(
  parameter = coef_names,
  mean = laplace_mode,
  sd = laplace_sd,
  lower95 = laplace_mode - 1.96 * laplace_sd,
  upper95 = laplace_mode + 1.96 * laplace_sd
)

m1_laplace_summary

# compile models
mod1 <- cmdstan_model("model1_baseline.stan", force_recompile = TRUE)
mod2 <- cmdstan_model("model2_hierarchical.stan", force_recompile = TRUE)

# model 1 hmc
stan_data_m1 <- list(
  N = N,
  K = K,
  X = X,
  y = as.integer(y)
)

fit1 <- mod1$sample(
  data = stan_data_m1,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  adapt_delta = 0.95,
  max_treedepth = 12,
  refresh = 100
)

fit1$summary(c("alpha", paste0("beta[", 1:K, "]")))

fit1_diag <- fit1$diagnostic_summary()
fit1_diag

m1_raw <- fit1$summary(c("alpha", paste0("beta[", 1:K, "]")))

m1_hmc_summary <- data.frame(
  parameter = coef_names,
  mean = m1_raw$mean,
  sd = m1_raw$sd,
  lower95 = m1_raw$q5,
  upper95 = m1_raw$q95,
  rhat = m1_raw$rhat,
  ess_bulk = m1_raw$ess_bulk,
  ess_tail = m1_raw$ess_tail
)

m1_hmc_summary

m1_compare <- data.frame(
  parameter = coef_names,
  laplace_mean = m1_laplace_summary$mean,
  laplace_sd = m1_laplace_summary$sd,
  laplace_lower95 = m1_laplace_summary$lower95,
  laplace_upper95 = m1_laplace_summary$upper95,
  hmc_mean = m1_hmc_summary$mean,
  hmc_sd = m1_hmc_summary$sd,
  hmc_lower95 = m1_hmc_summary$lower95,
  hmc_upper95 = m1_hmc_summary$upper95
)

m1_compare

# model 2 hmc
stan_data_m2 <- list(
  N = N,
  K = K,
  J = J,
  X = X,
  month = as.integer(month),
  y = as.integer(y)
)

fit2 <- mod2$sample(
  data = stan_data_m2,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  adapt_delta = 0.97,
  max_treedepth = 12,
  refresh = 100
)

fit2$summary(c("mu_alpha", "sigma_alpha", paste0("beta[", 1:K, "]")))

fit2_diag <- fit2$diagnostic_summary()
fit2_diag

m2_raw <- fit2$summary(c("mu_alpha", "sigma_alpha", paste0("beta[", 1:K, "]")))

m2_main_summary <- data.frame(
  parameter = c(
    "mu_alpha",
    "sigma_alpha",
    "PageValues_std",
    "BounceRates_std",
    "ExitRates_std",
    "ProductRelated_Duration_std",
    "ReturningVisitor",
    "Weekend"
  ),
  mean = m2_raw$mean,
  sd = m2_raw$sd,
  lower95 = m2_raw$q5,
  upper95 = m2_raw$q95,
  rhat = m2_raw$rhat,
  ess_bulk = m2_raw$ess_bulk,
  ess_tail = m2_raw$ess_tail
)

m2_main_summary

m2_month_raw <- fit2$summary(paste0("alpha[", 1:J, "]"))

m2_month_summary <- data.frame(
  Month = month_key$Month,
  MonthIndex = month_key$MonthIndex,
  mean = m2_month_raw$mean,
  sd = m2_month_raw$sd,
  lower95 = m2_month_raw$q5,
  upper95 = m2_month_raw$q95
)

m2_month_summary

# posterior predictive
yrep1 <- fit1$draws("y_rep")
yrep2 <- fit2$draws("y_rep")

yrep1_mat <- posterior::as_draws_matrix(yrep1)
yrep2_mat <- posterior::as_draws_matrix(yrep2)

yrep1_mat <- yrep1_mat[, grepl("^y_rep\\[", colnames(yrep1_mat)), drop = FALSE]
yrep2_mat <- yrep2_mat[, grepl("^y_rep\\[", colnames(yrep2_mat)), drop = FALSE]

ppc_overall <- data.frame(
  model = c("Model1", "Model2"),
  observed = mean(y),
  pred_mean = c(mean(rowMeans(yrep1_mat)), mean(rowMeans(yrep2_mat))),
  pred_lower95 = c(
    quantile(rowMeans(yrep1_mat), 0.025),
    quantile(rowMeans(yrep2_mat), 0.025)
  ),
  pred_upper95 = c(
    quantile(rowMeans(yrep1_mat), 0.975),
    quantile(rowMeans(yrep2_mat), 0.975)
  )
)

ppc_overall

ppc_month_model1 <- data.frame(
  Month = month_key$Month,
  MonthIndex = month_key$MonthIndex,
  observed = sapply(1:J, function(j) mean(y[month == j])),
  pred_mean = sapply(1:J, function(j) mean(rowMeans(yrep1_mat[, month == j, drop = FALSE]))),
  pred_lower95 = sapply(1:J, function(j) quantile(rowMeans(yrep1_mat[, month == j, drop = FALSE]), 0.025)),
  pred_upper95 = sapply(1:J, function(j) quantile(rowMeans(yrep1_mat[, month == j, drop = FALSE]), 0.975))
)

ppc_month_model2 <- data.frame(
  Month = month_key$Month,
  MonthIndex = month_key$MonthIndex,
  observed = sapply(1:J, function(j) mean(y[month == j])),
  pred_mean = sapply(1:J, function(j) mean(rowMeans(yrep2_mat[, month == j, drop = FALSE]))),
  pred_lower95 = sapply(1:J, function(j) quantile(rowMeans(yrep2_mat[, month == j, drop = FALSE]), 0.025)),
  pred_upper95 = sapply(1:J, function(j) quantile(rowMeans(yrep2_mat[, month == j, drop = FALSE]), 0.975))
)

ppc_month_model1
ppc_month_model2

plot(
  ppc_month_model1$MonthIndex,
  ppc_month_model1$observed,
  type = "b",
  pch = 16,
  xaxt = "n",
  ylim = range(c(
    ppc_month_model1$pred_lower95,
    ppc_month_model1$pred_upper95,
    ppc_month_model1$observed
  )),
  xlab = "Month",
  ylab = "Purchase rate",
  main = "ppc model 1"
)
axis(1, at = ppc_month_model1$MonthIndex, labels = ppc_month_model1$Month)
arrows(
  ppc_month_model1$MonthIndex,
  ppc_month_model1$pred_lower95,
  ppc_month_model1$MonthIndex,
  ppc_month_model1$pred_upper95,
  angle = 90,
  code = 3,
  length = 0.05,
  col = "blue"
)
points(ppc_month_model1$MonthIndex, ppc_month_model1$pred_mean, col = "blue", pch = 19)

plot(
  ppc_month_model2$MonthIndex,
  ppc_month_model2$observed,
  type = "b",
  pch = 16,
  xaxt = "n",
  ylim = range(c(
    ppc_month_model2$pred_lower95,
    ppc_month_model2$pred_upper95,
    ppc_month_model2$observed
  )),
  xlab = "Month",
  ylab = "Purchase rate",
  main = "ppc model 2"
)
axis(1, at = ppc_month_model2$MonthIndex, labels = ppc_month_model2$Month)
arrows(
  ppc_month_model2$MonthIndex,
  ppc_month_model2$pred_lower95,
  ppc_month_model2$MonthIndex,
  ppc_month_model2$pred_upper95,
  angle = 90,
  code = 3,
  length = 0.05,
  col = "blue"
)
points(ppc_month_model2$MonthIndex, ppc_month_model2$pred_mean, col = "blue", pch = 19)

# correctness check
true_alpha <- -1
true_beta <- c(0.8, -0.5, -0.8, 0.4, -0.3, 0.2)

eta_sim <- true_alpha + X %*% true_beta
p_sim <- plogis(eta_sim)
y_sim <- rbinom(N, 1, p_sim)

stan_data_sim <- list(
  N = N,
  K = K,
  X = X,
  y = as.integer(y_sim)
)

fit_sim <- mod1$sample(
  data = stan_data_sim,
  seed = 321,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  adapt_delta = 0.95,
  max_treedepth = 12,
  refresh = 100
)

fit_sim$summary(c("alpha", paste0("beta[", 1:K, "]")))

sim_raw <- fit_sim$summary(c("alpha", paste0("beta[", 1:K, "]")))

sim_summary <- data.frame(
  parameter = coef_names,
  true_value = c(true_alpha, true_beta),
  post_mean = sim_raw$mean,
  lower95 = sim_raw$q5,
  upper95 = sim_raw$q95
)

sim_summary




