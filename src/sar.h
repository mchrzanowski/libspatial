#pragma once
#include <armadillo>

class SAR {

public:
  SAR(const arma::colvec &y);
  virtual ~SAR();
  virtual void solve() = 0;

  const arma::colvec & get_beta();
  double get_rho();
  double get_sigma_sq();

protected:
  arma::colvec beta;
  const arma::colvec &y;
  double rho, sigma_sq;
};
