
data {
  int<lower=0> G;            // number of age groups
  int<lower=0> T;            // number of days
  int<lower=0> V;            // number of days since the vaccination started
  real vacc[V];              // proportion of vaccinated people
  int<lower=0> total[T, G];  // number of hospitalizations per day by age group
  real cases_total[T];       // number of confirmed cases per day
  int<lower=0> icu[T, G];    // number of icu admissions per day by age group
  matrix[T, G] male;         // proportion of male hospitalizations per day by age group
  matrix[T, G] cases_male;   // proportion of male cases per day by age group
  vector[2] zeros;           // initial mean for multivariate normal
  matrix[2, 2] ident;        // initial covariance matrix for multivariate normal
  real offset[G];            // log offset for Poisson (number of cases per 10000 people)

}

parameters {
  real alpha[T, G];       // dynamic baseline for total hospitalizations model (THM)
  real beta[G];           // proportion of male cases coefficient THM

  real<lower=0> W1;       // sd random walk for common random walk THM
  real<lower=0> W2[6];    // sd for alpha
  real<lower=0> W3;       // sd for beta

  real epsilon[T, G];     // dynamic baseline for ICU model (ICUM)
  real psi[G];            // proportion of male hospitalizations coefficient ICUM

  real<lower=0> W4;       // sd random walk for common random walk ICUM
  real<lower=0> W5;       // sd for epsilon
  real<lower=0> W6;       // sd for psi

  vector[2] mu[T];        // common random walk

  real nu1;               // shared mean for beta THM
  real nu2;               // shared mean for psi ICUM

  real delta;             // proportion of vaccinated people coefficient THM

  real<lower=0, upper=1> rho[G];   // transfer function coefficient
  real gamma[G];                   // transfer function coefficient

  real<lower=-1, upper=1> cor;     // correlation between random walks (mu)

  real eta[G];                     // hurdle model probability parameter
  real chi[G];                     // hurdle model probability parameter


}

transformed parameters{
  real<lower=0> lambda[T, G];          // rate parameter THM
  real<lower=0> theta[T, G];           // rate parameter ICUM
  matrix[T, G] u;                      // transfer function
  cov_matrix[2] W;                     // covariance matrix random walks
  real<lower=0, upper=1> probs[T, G];  // probability hurdle model


  // Transfer function definition
  for(j in 1:G) {
    u[1, j] = cases_total[1] * gamma[j];
  }
  for(i in 2:(T)) {
    for(j in 1:G){
      u[i, j] = cases_total[i] * gamma[j] + rho[j] * u[i - 1, j];
    }
  }

  // Rate parameter THM
  for(i in 1:(T - V)) {
    for(j in 1:G) {
      lambda[i, j] = exp(alpha[i, j] + beta[j] * cases_male[i, j] + offset[j] + u[i,j]);
    }
  }
  // Rate parameter with vaccination effect THM
  for(i in (T - V + 1):T) {
    for(j in 1:G) {
      lambda[i, j] = exp(alpha[i, j] + beta[j] * cases_male[i, j] + delta * vacc[i - (T - V)] + offset[j] + u[i , j]);
    }
  }

  // Rate parameter ICUM
  for(i in 1:(T)) {
    for(j in 1:G) {
      theta[i, j] = exp(epsilon[i, j] + psi[j] * male[i, j] + offset[j]);
    }
  }


  // Covariance function definition
  W[1, 1] = W1^2;
  W[1, 2] = cor * W1 * W4;
  W[2, 1] = W[1, 2];
  W[2, 2] = W4^2;

  // Probability Hurdle model
  for(j in 1:G) {
    probs[1, j] = inv_logit(eta[j]);
  }

  for(i in 2:T) {
    for(j in 1:G) {
      probs[i, j] = inv_logit(eta[j] + chi[j] * total[i - 1, j]);
    }
  }


}

model {

// Prior definition

  W1 ~ cauchy(0, 1);
  W2 ~ cauchy(0, 1);
  W3 ~ cauchy(0, 1);
  W4 ~ cauchy(0, 1);
  W5 ~ cauchy(0, 1);
  W6 ~ cauchy(0, 1);
  delta ~ normal(0, 1);
  eta ~ normal(0, 1);
  chi ~ normal(0, 1);
  gamma ~ normal(0, 10);
  cor ~ uniform(-1, 1);
  rho ~ uniform(0, 1);
  mu[1] ~ multi_normal(zeros, ident);
  nu1 ~ normal(0, 1);
  nu2 ~ normal(0, 1);

  // Random walk
  for(i in 2:T){
    mu[i] ~ multi_normal(mu[i - 1], W);

  }

  // Intercepts
  for(i in 1:(T)) {
    for(j in 1:G) {
      alpha[i, j] ~ normal(mu[i, 1], W2[j] ^ 2);
      epsilon[i, j] ~ normal(mu[i, 2], W5 ^ 2);
    }
  }

  // Proportion of male cases/hospitalizations coefficients
  for(j in 1:G) {
    beta[j] ~ normal(nu1, W3 ^ 2);
    psi[j] ~ normal(nu2, W6 ^ 2);
  }

  // THM
  for(i in 1:T) {
    for(j in 1:G){
      total[i, j] ~ poisson(lambda[i,j]);
    }
  }


  // ICUM
  for(i in 1:T) {
    for(j in 1:G){
      if (icu[i, j] == 0)
      1 ~ bernoulli(probs[i, j]);
      else {
        0 ~ bernoulli(probs[i, j]);
        icu[i, j] ~ poisson(theta[i,j]) T[1, ];
      }
    }
  }

}

generated quantities {
  int<lower=0> yfit[T, G];
  int<lower=0> icufit[T, G];
  int w;  // temporary variable
  real log_lik1[T, G];


  for(i in 1:T) {
    for(j in 1:G) {
        yfit[i, j] = poisson_rng(lambda[i, j]);
    }
  }


  for (i in 1:T) {
    for(j in 1:G) {
      if (bernoulli_rng(probs[i, j])) {
        icufit[i, j] = 0;
        log_lik1[i, j] = log(probs[i, j]) + poisson_lpmf(total[i,j] | lambda[i, j]);
      } else {
        // use a while loop because Stan doesn't have truncated RNGs
        w = poisson_rng(theta[i, j]);
        while (w == 0)
        w = poisson_rng(theta[i, j]);
        icufit[i, j] = w;
        log_lik1[i, j] = log1m(probs[i, j]) + poisson_lpmf(icu[i,j] | theta[i, j]) - poisson_lccdf(0 | theta[i, j]) + poisson_lpmf(total[i,j] | lambda[i, j]);
      }
    }
  }


}
