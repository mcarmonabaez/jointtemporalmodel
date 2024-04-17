options(mc.cores = parallel::detectCores())
library(rstan)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)


# Code to run the model ---------------------------------------------------


dat_list <- readRDS(file = 'dat_list.rds')
start_time <- Sys.time()
res_stan <- rstan::stan('joint_model.stan',
                        data = dat_list, chains = 2, iter = 25000,
                        warmup = 12500,
                        thin = 4,
                        pars = c('W1', 'W2', 'W3',
                                 'W4', 'W5', 'W6',
                                 'mu',
                                 'nu1', 'nu2',
                                 'lambda', 'theta',
                                 'alpha',
                                 'beta',
                                 'rho', 'gamma',
                                 'epsilon', 'psi',
                                 'delta', 'cor',
                                 'probs',
                                 'chi', 'eta',
                                 'yfit',
                                 'icufit',
                                 'log_lik1'
                        ), init = "random", init_r = 0.70,
                        control = list(adapt_delta = 0.85, max_treedepth = 15),
                        include = TRUE)

end_time <- Sys.time()
end_time - start_time
gc()


# Plot results ------------------------------------------------------------

summ_sims <- summary(res_stan)$summary %>%
  as_tibble() %>%
  mutate(param = rownames(summary(res_stan)$summary))


## Rate parameter of hospitalizations model
sims_lambda <- summ_sims |>
  filter(stringr::str_detect(param, 'lambda')) |>
  separate(param, into = c('time', 'age_group'), sep = ',') |>
  mutate(time = as.numeric(gsub("\\D", "", time)),
         age_group = as.numeric(gsub("\\D", "", age_group)))

sims_lambda |>
  ggplot(aes(time, mean, ymin = `2.5%`, ymax = `97.5%`)) +
  geom_line() +
  geom_ribbon(alpha = 0.3) +
  facet_wrap(~age_group)
