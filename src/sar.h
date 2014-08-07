#pragma once
#include <armadillo>

class SAR {

public:
  SAR(const arma::colvec &y, arma::uword m, arma::uword n);
  virtual ~SAR();
  virtual void solve() = 0;

  const arma::colvec & get_beta();
  double get_mu();
  double get_rho();
  double get_sigma_sq();

protected:
  virtual double rho_ll(double rho_hat) = 0;
  virtual double log_likelihood() = 0;
  
  void calculate_rho();
  void get_bound(double &x, const double multiplier);
  double golden_section_search(double a, double b, double c,
                                double tol=sqrt(arma::datum::eps));
  double bisection(double l, double u, double tol=sqrt(arma::datum::eps));

  arma::colvec y;
  const arma::uword m, n;
  const double mu;
  arma::colvec beta;
  double rho, sigma_sq;
};
