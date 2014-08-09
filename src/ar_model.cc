#include "ar_model.h"
#include <assert.h>
#include <cmath>

using namespace arma;

ARModel::ARModel(const colvec &y, const mat &X, const sp_mat &W) :
  y(y), W(W), X(X), m(X.n_rows), n(X.n_cols) {}

ARModel::~ARModel() {}

const colvec &
ARModel::get_beta(){
  return beta;
}

double
ARModel::get_rho(){
  return rho;
}

double
ARModel::get_sigma_sq(){
  return sigma_sq;
}

/* from: http://web.stanford.edu/class/ee364a/lectures/problems.pdf */
double
ARModel::bisection(double l, double u, double tol) {
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
ARModel::golden_section_search(double a, double b, double c, double tol) {
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
  
  double ll_x, ll_b;
  if (gss_cache.find(x) == gss_cache.end()){
    ll_x = rho_ll(x);
    gss_cache[x] = ll_x;
  }
  else {
    ll_x = gss_cache[x];
  }
  if (gss_cache.find(b) == gss_cache.end()){
    ll_b = rho_ll(b);
    gss_cache[b] = ll_b;
  }
  else {
    ll_b = gss_cache[b];
  }
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
ARModel::get_bound(double &x, const double multiplier){
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
ARModel::calculate_rho(){
  const double lower = -1 + datum::eps, upper = 1 - datum::eps;
  const double theta = 2 / (3 + sqrt(5));
  const double avg = (1 - theta) * lower + theta * upper;
  rho = golden_section_search(lower, avg, upper);
}
