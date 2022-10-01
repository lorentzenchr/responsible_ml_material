#==============================================================================
# Tuning a partly additive LightGBM model for French MTPL data
# Idea: Let car features interact freely, while other features are kept additive
# 
# Rerunning takes long; results might depend on seed
#==============================================================================

# Rerunning takes long; results might depend on seed

library(tidyverse)
library(splitTools)
library(lightgbm)

# Reload data and model-related objects
# main <- "french_motor"
main <- file.path("r", "french_motor")
load(file.path(main, "intro.RData"))

# Data interface of LightGBM
dtrain <- lgb.Dataset(
  data.matrix(train[x]), 
  label = train[[y]], 
  weight = train[[w]],
  params = list(feature_pre_filter = FALSE)
)

# Build interaction constraint vector
x_veh <- x[startsWith(x, "Veh")]
ic <- c(
  list(x_veh),
  as.list(setdiff(x, x_veh))
)
ic

# STEP 2: Select learning rate so that optimal number of rounds by early stopping is
# somewhere between 100 and 1000

params <- list(
  learning_rate = 0.05,  # play with this value
  objective = "poisson",
  interaction_constraints = ic,
  num_threads = 7  # choose number of threads in line with your system
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
# We add interaction constraints later as a list element can be quite a mess in a 
# data.frame
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
  num_threads = 7,
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
grid_file <- file.path(main, "grid_partly_additive_lgb.rds")

pb <- txtProgressBar(0, grid_size, style = 3)
for (i in seq_len(grid_size)) {
  params <- as.list(grid[i, -(1:2)])
  params$interaction_constraints <- ic
  
  cvm <- lgb.cv(
    params = params,
    data = dtrain,
    nrounds = 5000,
    folds = folds,
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
grid <- readRDS(grid_file) %>% 
  arrange(score)

# Show some strong parameter combinations
grid %>% 
  select_if(function(x) n_distinct(x) >= 2L) %>% 
  head()

# Fit model on best parameter combination
params <- as.list(grid[i, -(1:2)])
params$interaction_constraints <- ic

fit_lgb_part_add <- lgb.train(
  params = params, 
  data = dtrain, 
  nrounds = grid[1, "iteration"]
)

library(flashlight)
library(MetricsWeighted)

# Quick evaluation
fl_lgb_part_add <- flashlight(
  model = fit_lgb_part_add, 
  label = "Partly additive", 
  data = test, 
  y = y, 
  w = w, 
  metrics = list(`Relative deviance reduction` = r_squared_poisson),
  predict_function = function(fit, X) predict(fit, data.matrix(X[x])),
  linkinv = log
)

light_performance(fl_lgb_part_add) # 0.0484
light_ice(fl_lgb_part_add, "DrivAge", center = "middle") %>% 
  plot()
light_ice(fl_lgb_part_add, "VehAge", center = "middle") %>% 
  plot()

