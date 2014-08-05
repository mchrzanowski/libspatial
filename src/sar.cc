#include "sar.h"

SAR::SAR(const arma::colvec &y) : y(y) {}

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
