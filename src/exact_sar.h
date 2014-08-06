#include "sar.h"

class ExactSAR : public SAR {
public:
  ExactSAR(const arma::mat &W, const arma::mat &X, const arma::colvec &y);
  void solve();

protected:
  double log_likelihood(double rho_hat);

private:
  const arma::mat &W;
  const arma::mat &X, IX;
  arma::colvec eigs;
};
