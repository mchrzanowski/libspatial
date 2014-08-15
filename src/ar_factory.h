#pragma once
#include <armadillo>
#include "car.h"
#include "sar.h"

/* provide the y vector, X design matrix, W neighborhood matrix, and whether
we should compute log_det(I - \rhoW) during training or just an approximation */
CAR * get_CAR_solver(const arma::colvec &y, const arma::mat &X,
                      const arma::sp_mat &W, bool exact=false);

/* provide the y vector, X design matrix, W neighborhood matrix, and whether
we should compute log_det(I - \rhoW) during training or just an approximation */
SAR * get_SAR_solver(const arma::colvec &y, const arma::mat &X,
                      const arma::sp_mat &W, bool exact=false);
