#include "car.h"

class ExactCAR : public CAR {
public:
  ExactCAR(const arma::colvec &y, const arma::mat &X, const arma::sp_mat &W);
  double log_likelihood();

protected:
  double rho_ll(double rho_hat);

private:
  const arma::colvec eigs;
};
