#include "sar.h"
#include <assert.h>
#include <cmath>

using namespace arma;

SAR::SAR(const colvec &y, const mat &X, const sp_mat &W) : 
  y(y), W(W), X(X),
  IX(speye<sp_mat>(X.n_rows, X.n_rows) - X * pinv(X.t() * X) * X.t()),
  m(X.n_rows), n(X.n_cols) {}

SAR::~SAR() {}

const colvec &
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
SAR::solve(){
  calculate_rho();
  const sp_mat A = speye<sp_mat>(W.n_rows, W.n_cols) - rho * W;
  const colvec A_y = A * y;
  if (! arma::solve(beta, X.t() * X, X.t() * A_y) || ! is_finite(beta)){
    beta = zeros<colvec>(X.n_cols);  
  }
  const colvec IX_A_y = IX * A_y;
  sigma_sq = dot(IX_A_y, IX_A_y) / m;
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
  const double lower = 0, upper = 1 - datum::eps;
  const double theta = 2 / (3 + sqrt(5));
  const double avg = (1 - theta) * lower + theta * upper;
  rho = golden_section_search(lower, avg, upper);
}
