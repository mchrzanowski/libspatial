#include "sar.h"

class ExactSAR : public SAR {
public:
  ExactSAR(const arma::mat &W, const arma::mat &X, const arma::colvec &y);
  void solve();

protected:
  double rho_ll(double rho_hat);
  double log_likelihood();

  const arma::mat &W;
  const arma::mat &X, IX;
  const arma::colvec eigs;
};
