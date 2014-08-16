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

  const arma::mat XT, XT_X, IX;
  const arma::colvec W_y, XT_W_y, XT_y;
};
