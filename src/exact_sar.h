#include "sar.h"

class ExactSAR : public SAR {
public:
  ExactSAR(const arma::colvec &y, const arma::mat &X, const arma::sp_mat &W);

protected:
  double calc_log_det(double rho_hat);

private:
  const arma::colvec eigs;
};
