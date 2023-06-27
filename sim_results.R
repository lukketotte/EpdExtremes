library(tidyverse)
library(gridExtra)
library(grid)

## load result files
setwd('C:/Users/aleen962/Dropbox/PhD/Forskning/Power exponential dist extremes/EpdExtremes/results')
# file_names <- list.files(pattern="list")
# for(i in 1:length(file_names)) assign(gsub("-|.csv", "", file_names[[i]]), read_csv(file = file_names[[i]], col_names = c('lambda', 'nu', 'beta')))
# rm(file_names)

lapply(list.files(pattern="list"), load, .GlobalEnv)

bias_fun <- function(estimates, true_val){
  return(colMeans(estimates - true_val))
}

rmse_fun <- function(estimates, true_val){
  return(sqrt(colMeans((true_val - estimates)^2)))
}

true_beta <- seq(0.2, 0.9, by = 0.1)
true_lambda <- 0.5
bias_res_d5 <- bias_res_d10 <- bias_res_d15 <- tibble('lambda' = numeric(8),
                                                      'nu' = numeric(8),
                                                      'beta' = numeric(8),
                                                      'beta_val' = numeric(8),
                                                      'dim' = numeric(8)
                                                      )
rmse_res_d5 <- rmse_res_d10 <- rmse_res_d15 <- tibble('lambda' = numeric(8),
                                                      'nu' = numeric(8),
                                                      'beta' = numeric(8),
                                                      'beta_val' = numeric(8),
                                                      'dim' = numeric(8)
                                                      )
for(i in 1:8){
  # dat_d5 <- dim5_lambda1.0_list[[i]]
  # dat_d10 <- dim10_lambda1.0_list[[i]]
  # dat_d15 <- dim15_lambda1.0_list[[i]]
  
  dat_d5 <- dim5_lambda0.5_list[[i]]
  dat_d10 <- dim10_lambda0.5_list[[i]]
  dat_d15 <- dim15_lambda0.5_list[[i]]
  
  bias_res_d5[i,1] <- bias_fun(dat_d5[1], true_lambda)
  bias_res_d5[i,2] <- bias_fun(dat_d5[2], 1)
  bias_res_d5[i,3] <- bias_fun(dat_d5[3], true_beta[i])
  bias_res_d5[i,4] <- true_beta[i]

  bias_res_d10[i,1] <- bias_fun(dat_d10[1], true_lambda)
  bias_res_d10[i,2] <- bias_fun(dat_d10[2], 1)
  bias_res_d10[i,3] <- bias_fun(dat_d10[3], true_beta[i])
  bias_res_d10[i,4] <- true_beta[i]

  bias_res_d15[i,1] <- bias_fun(dat_d15[1], true_lambda)
  bias_res_d15[i,2] <- bias_fun(dat_d15[2], 1)
  bias_res_d15[i,3] <- bias_fun(dat_d15[3], true_beta[i])
  bias_res_d15[i,4] <- true_beta[i]

  rmse_res_d5[i,1] <- rmse_fun(dat_d5[1], true_lambda)
  rmse_res_d5[i,2] <- rmse_fun(dat_d5[2], 1)
  rmse_res_d5[i,3] <- rmse_fun(dat_d5[3], true_beta[i])
  rmse_res_d5[i,4] <- true_beta[i]

  rmse_res_d10[i,1] <- rmse_fun(dat_d10[1], true_lambda)
  rmse_res_d10[i,2] <- rmse_fun(dat_d10[2], 1)
  rmse_res_d10[i,3] <- rmse_fun(dat_d10[3], true_beta[i])
  rmse_res_d10[i,4] <- true_beta[i]

  rmse_res_d15[i,1] <- rmse_fun(dat_d15[1], true_lambda)
  rmse_res_d15[i,2] <- rmse_fun(dat_d15[2], 1)
  rmse_res_d15[i,3] <- rmse_fun(dat_d15[3], true_beta[i])
  rmse_res_d15[i,4] <- true_beta[i]
}
bias_res <- bind_rows(bias_res_d5,
                      bias_res_d10,
                      bias_res_d15
                      )
rmse_res <- bind_rows(rmse_res_d5,
                      rmse_res_d10,
                      rmse_res_d15
                      )
bias_res$dim <- c(rep('d = 5',8), rep('d = 10',8), rep('d = 15',8))
rmse_res$dim <- c(rep('d = 5',8), rep('d = 10',8), rep('d = 15',8))

p_bias_lam05 <- bias_res %>% 
  pivot_longer(cols = 1:3, names_to = 'param', values_to = 'bias') %>% 
  mutate(param = as_factor(param), dim = as_factor(dim)) %>% 
  ggplot() + 
  geom_line(aes(x = beta_val, bias, color = param), linewidth = 1) +
  geom_point(aes(x = beta_val, bias, shape = param, color = param), size = 3) +
  scale_shape_manual(values = c("lambda" = 0, "nu" = 1, "beta" = 2)) +
  # scale_linetype_manual(values = c("lambda" = 4, "nu" = 2, "beta" = 1)) +
  scale_color_manual(values = c("lambda" = '#CC79A7', "nu" = '#009E73', "beta" = '#666666')) + 
  geom_hline(yintercept = 0, linetype = 3) +
  theme_bw(base_size = 16) + 
  ggtitle(expression(paste(lambda, ' = 0.5', sep = ''))) +
  labs(x = expression(beta),
       y = 'Bias',
       color = 'Parameter',
       shape = 'Parameter') +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # legend.position = 'none',
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 0, hjust = .5, vjust = .7),
        strip.background = element_blank(),
        strip.placement = "outside",
        strip.text.x = element_text(size = 14),
        panel.border = element_rect(colour = 'black', fill = NA, linewidth = 0.01)
        ) +
  facet_wrap(~dim)
p_bias_lam05
p_bias_lam10
grid.arrange(p_bias_lam05, p_bias_lam10, nrow = 2)


p_rmse_lam05 <- rmse_res %>% 
  pivot_longer(cols = 1:3, names_to = 'param', values_to = 'rmse') %>% 
  mutate(param = as_factor(param), dim = as_factor(dim)) %>% 
  ggplot() + 
  geom_line(aes(x = beta_val, rmse, color = param), linewidth = 1) +
  geom_point(aes(x = beta_val, rmse, shape = param, color = param), size = 3) +
  scale_shape_manual(values = c("lambda" = 0, "nu" = 1, "beta" = 2)) +
  # scale_linetype_manual(values = c("lambda" = 4, "nu" = 2, "beta" = 1)) +
  scale_color_manual(values = c("lambda" = '#CC79A7', "nu" = '#009E73', "beta" = '#666666')) + 
  geom_hline(yintercept = 0, linetype = 3) +
  theme_bw(base_size = 16) + 
  ggtitle(expression(paste(lambda, ' = 0.5', sep = ''))) +
  labs(x = expression(beta),
       y = 'RMSE',
       color = 'Parameter',
       shape = 'Parameter') +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # legend.position = 'none',
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 0, hjust = .5, vjust = .7),
        strip.background = element_blank(),
        strip.placement = "outside",
        strip.text.x = element_text(size = 14),
        panel.border = element_rect(colour = 'black', fill = NA, linewidth = 0.01)
        ) +
  facet_wrap(~dim)
p_rmse_lam05
p_rmse_lam10
grid.arrange(p_rmse_lam05, p_rmse_lam10, nrow = 2)


