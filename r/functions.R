# Mean Poisson deviance and corresponding pseudo R-squared
poisson_scorer <- function(
    model, X, y, w, reference_mean = weighted.mean(y, w), pred_fun = predict, ...
  ) {
  deviance_0 <- deviance_poisson(y, rep(reference_mean, nrow(X)), w = w)
  preds <- as.matrix(pred_fun(model, X, ...))
  deviances <- numeric(ncol(preds))
  names(deviances) <- colnames(preds)
  for (i in 1:ncol(preds)) {
    deviances[i] <- deviance_poisson(y, preds[, i], w = w)
  }
  data.frame(mean_deviance = deviances, Pseudo_R2 = 1 - deviances / deviance_0)
}
