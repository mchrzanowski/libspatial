#include "car.h"

class ExactCAR : public CAR {
public:
  ExactCAR(const arma::colvec &y, const arma::mat &X, const arma::sp_mat &W);

protected:
  double calc_log_det(double rho_hat);

private:
  const arma::colvec eigs;
  const arma::mat X_T_W;
};
