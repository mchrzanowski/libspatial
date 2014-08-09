#include "sar.h"

class ChebyshevSAR : public SAR {
public:
  ChebyshevSAR(const arma::colvec &y, const arma::mat &X,
                const arma::sp_mat &W);
  double log_likelihood();

protected:
  double rho_ll(double rho_hat);

private:
  arma::mat cheby_poly_coeffs;
};
