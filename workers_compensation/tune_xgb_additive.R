#======================================================================================
# Code used to find the parameter combination for the additive XGBoost model of
# Exercise 3i of Chapter 3: Improving Explainability
#======================================================================================

dtrain <- xgb.DMatrix(data.matrix(X_train), label = y_train)

# STEP 1: Model for the expected loss -> Gamma deviance

# STEP 2: Select learning rate to get reasonable number of trees by early stopping
params <- list(
  learning_rate = 0.1,  # play with this value
  objective = "reg:gamma",
  max_depth = 1, # this is the main point...
  tree_method = "hist",
  nthread = 7
)

cvm <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 5000,
  nfold = 5,
  early_stopping_rounds = 20,
  showsd = FALSE, 
  print_every_n = 10
)
cvm

# STEP 3: Random search for other parameters (some parameters are fixed)
grid <- expand.grid(
  iteration = NA,
  cv_score = NA,
  train_score = NA,
  learning_rate = 0.1,
  objective = "reg:gamma",
  max_depth = 1,
  colsample_bynode = c(0.8, 1),
  subsample = c(0.8, 1),
  reg_alpha = 0:3,
  reg_lambda = 0:3,
  tree_method = "hist",
  min_split_loss = c(0, 0.001),
  nthread = 7,
  stringsAsFactors = FALSE
)

# Full grid search or randomized search?
max_size <- 20
grid_size <- nrow(grid)
if (grid_size > max_size) {
  grid <- grid[sample(grid_size, max_size), ]
  grid_size <- max_size
}

# Loop over grid and fit XGB with five-fold CV and early stopping
grid_file <- file.path(main, "grid_xgb_additive.rds")

pb <- txtProgressBar(0, grid_size, style = 3)
for (i in seq_len(grid_size)) {  # i <- 1
  cvm <- xgb.cv(
    params = as.list(grid[i, -(1:3)]),
    data = dtrain,
    nrounds = 5000,
    nfold = 5,
    early_stopping_rounds = 20,
    verbose = 0
  )
  
  # Store optimal number of boosting rounds and both the training and the CV score
  grid[i, 1] <- cvm$best_iteration
  grid[i, 2:3] <- cvm$evaluation_log[, c(4, 2)][cvm$best_iteration]
  setTxtProgressBar(pb, i)
  
  # Save grid to survive hard crashs
  saveRDS(grid, file = grid_file)
}

# Load and sort grid
grid <- readRDS(grid_file) %>%
  arrange(cv_score)
head(grid)

# Fit final model with best parameter combination
with_seed(3089, 
  xgb_add <- xgb.train(
    params = as.list(grid[1, -(1:3)]), 
    data = dtrain, 
    nrounds = grid[1, "iteration"]
  )
)
