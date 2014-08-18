#pragma once
#include "ar_model.h"

class SAR : public ARModel {

public:
  SAR(const arma::colvec &y, const arma::mat &X, const arma::sp_mat &W);
  void solve();
  double log_likelihood();

protected:
  double rho_ll(double rho_hat);
  virtual double calc_log_det(double rho_hat) = 0;

  const arma::mat XT;
  const arma::sp_mat I;
};
