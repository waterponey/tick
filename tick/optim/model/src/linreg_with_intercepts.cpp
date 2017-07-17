// License: BSD 3 clause

#include "linreg_with_intercepts.h"

ModelLinRegWithIntercepts::ModelLinRegWithIntercepts(const SBaseArrayDouble2dPtr features,
                                                     const SArrayDoublePtr labels,
                                                     const bool fit_intercept,
                                                     const int n_threads)
    : ModelGeneralizedLinear(features, labels, fit_intercept, n_threads),
      ModelGeneralizedLinearWithIntercepts(features, labels, fit_intercept, n_threads),
      ModelLinReg(features, labels, fit_intercept, n_threads) {}

const char *ModelLinRegWithIntercepts::get_class_name() const {
  return "ModelLinRegWithIntercepts";
}

void ModelLinRegWithIntercepts::compute_lip_consts() {
  // TODO: when fit_intercept is true, this should be different
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = ArrayDouble(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      lip_consts[i] = features_norm_sq[i] + 1;
    }
  }
}
