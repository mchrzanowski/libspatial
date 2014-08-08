#include "sar.h"

class ExactSAR : public SAR {
public:
  ExactSAR(const arma::sp_mat &W, const arma::mat &X, const arma::colvec &y);
  double log_likelihood();

protected:
  double rho_ll(double rho_hat);

private:
  const arma::colvec eigs;
};
