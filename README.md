libspatial
=======

This is a software library meant to be a performance-competitive alternative to
the library [<tt>spdep</tt>](http://cran.r-project.org/web/packages/spdep/index.html).
Currently supported features include:

* solving SAR and CAR models
* creating sparse adjacency matrices for images

For more details, see [the preliminary paper about the library](https://www.dropbox.com/s/06rez668ncbzj7k/project.pdf).

###Dependencies
* <tt>g++</tt> (at least v. 4.8).
* <tt>armadillo</tt> (at least v. 4.300).
* <tt>CImg</tt> (at least v. 1.5.9).
* a linear algebra subroutine package supported by <tt>armadillo</tt> such as
  <tt>Intel MKL</tt>.
* Mac OS X (10.9) or Linux.

###Installation
To compile <tt>libspatial</tt> as well as the example file <tt>testing.cc</tt>,
follow the following steps:

1. You'll need modify the Makefile in the <tt>src</tt> directory to specify a
linear algebra library to link against. The Makefile currently uses the Intel
MKL as the linear algebra backend, but there are commented out lines for
supported OpenBLAS as well. Should you wish to use some other linear algebra
library, modify the Makefile accordingly. 

2. Run ```make```.

###Usage
Example usage is provided in <tt>testing.cc</tt>
