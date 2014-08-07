#include "sar.h"
#include <assert.h>
#include <cmath>

SAR::SAR(const arma::colvec &_y, arma::uword m, arma::uword n) : y(_y), m(m),
  n(n), mu(arma::mean(_y)) {
    y -= mu;
}

SAR::~SAR() {}

const arma::colvec &
SAR::get_beta(){
  return beta;
}

double
SAR::get_mu(){
  return mu;
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
    ll = rho_ll(t);
    if (! isnan(ll) && ! isinf(ll)){
      u = t;
    }
    else {
      l = t;
    }
  } while ((u - l) > tol);
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

  double ll_x = rho_ll(x);
  double ll_b = rho_ll(b);
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

void
SAR::get_bound(double &x, const double multiplier){
  double ll = rho_ll(x); 
  while (isinf(ll) || isnan(ll)) {
    if (x == 0){
      x = .01;
    }
    else {
      x *= multiplier;
    }
    ll = rho_ll(x);
  }
}

void
SAR::calculate_rho(){
  double lower = 0, upper = 1 - arma::datum::eps;
  get_bound(lower, 1.05);
  get_bound(upper, 0.95);

  double theta = 2 / (3 + sqrt(5));
  double avg = (1 - theta) * lower + theta * upper;
  rho = golden_section_search(lower, avg, upper);
}
