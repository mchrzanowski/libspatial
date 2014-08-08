#pragma once
#include <armadillo>

class SAR {

public:
  SAR(const arma::colvec &y, const arma::mat &X, const arma::sp_mat &W);
  virtual ~SAR();

  void solve();
  virtual double log_likelihood() = 0;

  const arma::colvec & get_beta();
  double get_rho();
  double get_sigma_sq();

protected:
  virtual double rho_ll(double rho_hat) = 0;
  void calculate_rho();
  void get_bound(double &x, const double multiplier);
  double golden_section_search(double a, double b, double c,
                                double tol=sqrt(arma::datum::eps));
  double bisection(double l, double u, double tol=sqrt(arma::datum::eps));

  const arma::colvec &y;
  const arma::sp_mat &W;
  const arma::mat &X, IX;
  const arma::uword m, n;

  arma::colvec beta;
  double rho, sigma_sq;
};
