#==============================================================================
# Tuning an additive LightGBM model for French MTPL data
# 
# Rerunning takes long; results might depend on seed
#==============================================================================

# Rerunning takes long; results might depend on seed
library(tidyverse)
library(splitTools)
library(lightgbm)

# Reload data and model-related objects
main <- "french_motor"
load(file.path(main, "intro.RData"))

# Data interface of LightGBM
dtrain <- lgb.Dataset(
  data.matrix(train[xvars]), 
  label = train$Freq, 
  weight = train$Exposure
)

# STEP 1: Model for expected claims frequency -> Poisson deviance

# STEP 2: Select learning rate so that optimal number of rounds by early stopping is
# somewhere between 100 and 1000
params <- list(
  objective = "poisson",
  learning_rate = 0.5,  # much higher than for a more flexibel model
  num_leaves = 2, # Or max_depth = 1
)

# Grouped cross-validation folds (hold-out indices)
folds <- create_folds(train$group_id, k = 5, type = "grouped", invert = TRUE)

# k-fold grouped cross-validation to see how many trees are required by early stopping
cvm <- lgb.cv(
  params = params,
  data = dtrain,
  nrounds = 5000,
  folds = folds,
  early_stopping_rounds = 20,
  eval_freq = 50
)
cvm

# STEP 3: Random search for other parameters (some parameters are fixed)
grid <- expand.grid(
  iteration = NA,
  score = NA,
  learning_rate = 0.5,
  objective = "poisson",
  num_leaves = 2,  # Or max_depth = 1
  lambda_l2 = c(0, 2.5, 5, 7.5),
  lambda_l1 = c(0, 2, 4),
  colsample_bynode = c(0.8, 1),
  bagging_fraction = c(0.8, 1),
  # min_data_in_leaf = c(20, 50, 100),
  # min_sum_hessian_in_leaf = c(0, 0.001, 0.1),
  poisson_max_delta_step = c(0.1, 0.7),  # only for Poisson objective
  stringsAsFactors = FALSE
)

# Full grid search or randomized search?
max_size <- 20
grid_size <- nrow(grid)
if (grid_size > max_size) {
  grid <- grid[sample(grid_size, max_size), ]
  grid_size <- max_size
}

# Loop over grid and fit LGB with grouped five-fold CV and early stopping
grid_file <- file.path(main, "grid_additive_lgb.rds")

pb <- txtProgressBar(0, grid_size, style = 3)
for (i in seq_len(grid_size)) {
  cvm <- lgb.cv(
    params = as.list(grid[i, -(1:2)]),
    data = dtrain,
    nrounds = 5000,
    folds = folds,
    # nfold = 5,  # for the ungrouped case
    early_stopping_rounds = 20,
    verbose = -1
  )
  
  # Store optimal number of boosting rounds and the cross-validation score
  grid[i, 1:2] <- as.list(cvm)[c("best_iter", "best_score")]
  setTxtProgressBar(pb, i)
  
  # Save grid to survive hard crashs
  saveRDS(grid, file = grid_file)
}

# Load grid
grid <- readRDS(grid_file) |> 
  arrange(score)

# Show some strong parameter combinations
grid |> 
  select_if(function(x) n_distinct(x) >= 2L) |> 
  head()

# Fit model on best parameter combination
fit_lgb_add <- lgb.train(
  params = as.list(grid[1, -(1:2)]), 
  data = dtrain, 
  nrounds = grid[1, "iteration"]
)
