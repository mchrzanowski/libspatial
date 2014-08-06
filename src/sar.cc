#include "sar.h"
#include <assert.h>
#include <cmath>

SAR::SAR(const arma::colvec &y, arma::uword m, arma::uword n) : y(y), m(m),
  n(n) {}

SAR::~SAR() {}

const arma::colvec &
SAR::get_beta(){
  return beta;
}

double
SAR::get_rho(){
  return rho;
}

double
SAR::get_sigma_sq(){
  return sigma_sq;
}

/* from: http://web.stanford.edu/class/ee364a/lectures/problems.pdf */
double
SAR::bisection(double l, double u, double tol) {
  assert(l < u);
  assert(tol > 0);
  
  double t, ll;
  do {
    t = (l + u) / 2;
    ll = log_likelihood(t);
    if (isnan(ll) || isinf(ll)){
      u = t;
    }
    else {
      l = t;
    }
  } while (u - l > tol);
  return t;
}

/* algo from: http://en.wikipedia.org/wiki/Golden_section_search */
double
SAR::golden_section_search(double a, double b, double c, double tol) {
  static const double phi = (1 + sqrt(5)) / 2;
  static const double resphi = 2 - phi;
  double x;
  if (c - b > b - a){
    x = b + resphi * (c - b);
  }
  else {
    x = b - resphi * (b - a);
  }
  if (abs(c - a) <= tol * (abs(b) + abs(x))){
    return (c + a) / 2; 
  }

  double ll_x = log_likelihood(x);
  double ll_b = log_likelihood(b);
  assert(ll_x != ll_b);
  if (ll_x < ll_b) {
    if (c - b > b - a) {
      return golden_section_search(b, x, c, tol);
    }
    else {
      return golden_section_search(a, x, b, tol);
    }
  }
  else {
    if (c - b > b - a) {
      return golden_section_search(a, b, x, tol);
    }
    else {
      return golden_section_search(x, b, c, tol);
    }
  }
}
