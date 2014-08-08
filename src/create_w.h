#include <armadillo>

using namespace arma;

/* creates a symmetric, sparse, row-stochastic neighborhood matrix.
  this matrix has the nice property that the absolute value of all its
  eigenvalues are less than or equal to 1. */
void
create_4_neighbor_W(uword points, uword img_width, sp_mat &W){
  W = zeros<sp_mat>(points, points);
  colvec d = zeros<colvec>(points);
  for (uword i = 0; i < img_width; i++){
    for (uword j = 0; j < img_width; j++){
      uword sourceIndx = j * img_width + i, destIndx;
      if (j + 1 < img_width){
        destIndx = (j + 1) * img_width + i;
        W(sourceIndx, destIndx) = 1;
        d[sourceIndx]++;
      } 
      if (i + 1 < img_width) {
        destIndx = j * img_width + i + 1;
        W(sourceIndx, destIndx) = 1;
        d[sourceIndx]++;
      }
      if (i > 0) {
        destIndx = j * img_width + i - 1;
        W(sourceIndx, destIndx) = 1;
        d[sourceIndx]++;
      }
      if (j > 0){
        destIndx = (j - 1) * img_width + i;
        W(sourceIndx, destIndx) = 1;
        d[sourceIndx]++;
      }
    }
  }
  const mat D = diagmat(1 / sqrt(d));
  W = D * W * D;
}
