#pragma once
#include <armadillo>

class ExactSAR {
public:
  ExactSAR(const arma::mat &W, const arma::mat &X, const arma::colvec &y);
  void solve();

private:
  double log_likelihood(double phi_hat);
  double golden_section_search(double a, double b, double c, double tol);

  const arma::mat &W, &X, IX;
  const arma::colvec &y;
  arma::colvec beta, eigs;
  double phi, sigma_sq;
};
