#pragma once
#include <armadillo>
#include <map>

class ARModel {
public:
  ARModel(const arma::colvec &y, const arma::mat &X, const arma::sp_mat &W);
  virtual ~ARModel();
  virtual void solve() = 0;
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
  const arma::mat &X;
  const arma::uword m, n;

  arma::colvec beta;
  double rho, sigma_sq;

private:
  std::map<double, double> gss_cache;
};
