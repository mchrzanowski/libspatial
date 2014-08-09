#include <armadillo>

using namespace arma;

/* creates a symmetric, sparse, neighborhood matrix.
  this matrix has the nice property that the absolute value of all its
  eigenvalues are less than or equal to 1. */
void
create_4_neighbor_W(uword points, uword img_height, sp_mat &W){
  W = zeros<sp_mat>(points, points);
  colvec d = zeros<colvec>(points);
  for (uword i = 0; i < img_height; i++){
    for (uword j = 0; j < img_height; j++){
      uword source = j * img_height + i, dest;
      if (j + 1 < img_height){
        dest = (j + 1) * img_height + i;
        W(source, dest) = 1;
        d[source]++;
      } 
      if (i + 1 < img_height) {
        dest = j * img_height + i + 1;
        W(source, dest) = 1;
        d[source]++;
      }
      if (i > 0) {
        dest = j * img_height + i - 1;
        W(source, dest) = 1;
        d[source]++;
      }
      if (j > 0){
        dest = (j - 1) * img_height + i;
        W(source, dest) = 1;
        d[source]++;
      }
    }
  }
  const mat D = diagmat(1 / sqrt(d));
  W = D * W * D;
}
