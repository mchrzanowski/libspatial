#pragma once
#include "ar_model.h"

class CAR : public ARModel {

public:
  CAR(const arma::colvec &y, const arma::mat &X, const arma::sp_mat &W);
  void solve();
  double log_likelihood() = 0;

protected:
  double rho_ll(double rho_hat) = 0;
};
