#pragma once
#include <armadillo>

class SAR {

public:
  SAR(const arma::colvec &y, arma::uword m, arma::uword n);
  virtual ~SAR();
  virtual void solve() = 0;

  const arma::colvec & get_beta();
  double get_rho();
  double get_sigma_sq();

protected:
  const arma::colvec &y;
  const arma::uword m, n;
  arma::colvec beta;
  double rho, sigma_sq;
};
