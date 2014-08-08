#include "sar.h"

class ChebyshevSAR : public SAR {
public:
  ChebyshevSAR(const arma::sp_mat &W, const arma::mat &X,
                const arma::colvec &y);
  double log_likelihood();

protected:
  double rho_ll(double rho_hat);

private:
  arma::mat cheby_poly_coeffs;
};
