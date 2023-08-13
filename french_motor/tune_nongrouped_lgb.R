#==============================================================================
# Ungrouped CV randomized search for the LightGBM model for French MTPL data
# 
# Rerunning takes long; results might depend on seed
#==============================================================================

library(withr)
library(tidyverse)
library(lightgbm)

# Reload data and model-related objects
main <- "french_motor"
# main <- file.path("r", "french_motor")
load(file.path(main, "intro.RData"))

# Random split
n <- nrow(prep)
with_seed(1, 
  ix <- sample(1:n, 0.8 * n)
)
train_u <- prep[ix, ]
test_u <- prep[-ix, ]

# Data interface of LGB
dtrain_u <- lgb.Dataset(
  data.matrix(train_u[x]),
  label = train_u[[y]],
  weight = train_u[[w]],
  params = list(feature_pre_filter = FALSE)
)

# STEP 1: Model for expected claims frequency -> Poisson deviance

# Step 2: select learning rate so that optimal number of rounds is
# somewhere between 100 and 1000
params <- list(
  learning_rate = 0.05,
  objective = "poisson",
  num_threads = 7  # choose number of threads in line with your system
)

# k-fold grouped cross-validation
cvm <- lgb.cv(
  params = params,
  data = dtrain_u,
  nrounds = 5000,
  nfold = 5,
  early_stopping_rounds = 20,
  eval_freq = 50
)
cvm

# Step 3: Grid search
grid <- expand.grid(
  iteration = NA,
  score = NA,
  learning_rate = 0.05,
  objective = "poisson",
  metric = "poisson",
  num_leaves = c(15, 31, 63),
  min_data_in_leaf = c(20, 50, 100),
  min_sum_hessian_in_leaf = c(0, 0.001, 0.1),
  colsample_bynode = c(0.8, 1),
  bagging_fraction = c(0.8, 1),
  lambda_l1 = c(0, 4),
  lambda_l2 = c(0, 2.5, 5, 7.5),
  # poisson_max_delta_step = c(0.1, 0.7),  # should try with Poisson objective
  num_threads = 7,
  stringsAsFactors = FALSE
)

# Grid search or randomized search if grid is too large
max_size <- 20
grid_size <- nrow(grid)
if (grid_size > max_size) {
  grid <- grid[sample(grid_size, max_size), ]
  grid_size <- max_size
}

# Loop over grid and fit LGB with five-fold CV and early stopping
grid_file <- file.path(main, "grid_nongrouped_lgb.rds")

pb <- txtProgressBar(0, grid_size, style = 3)
for (i in seq_len(grid_size)) { # i <- 1
  cvm <- lgb.cv(
    params = as.list(grid[i, -(1:2)]),
    data = dtrain_u,
    nrounds = 5000,
    nfold = 5,
    early_stopping_rounds = 20,
    verbose = -1
  )

  # Store result
  grid[i, 1:2] <- as.list(cvm)[c("best_iter", "best_score")]
  setTxtProgressBar(pb, i)

  # Save grid to survive hard crashs
  saveRDS(grid, file = grid_file)
}

# Load grid
grid <- readRDS(grid_file) %>% 
  arrange(score)
head(grid)

# Fit model on best parameter combination
fit_lgb <- lgb.train(
  params = as.list(grid[1, -(1:2)]), 
  data = dtrain_u, 
  nrounds = grid[1, "iteration"]
)
