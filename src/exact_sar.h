#include "sar.h"

class ExactSAR : public SAR {
public:
  ExactSAR(const arma::mat &W, const arma::mat &X, const arma::colvec &y);
  void solve();

private:
  double log_likelihood(double rho_hat);
  double golden_section_search(double a, double b, double c,
                                double tol=sqrt(arma::datum::eps));

  const arma::mat &W;
  const arma::mat &X, IX;
  arma::colvec eigs;
};
