#include "car.h"

class ChebyshevCAR : public CAR {
public:
  ChebyshevCAR(const arma::colvec &y, const arma::mat &X,
                const arma::sp_mat &W);

protected:
  double calc_log_det(double rho_hat);

private:
  arma::mat cheby_poly_coeffs;
};
