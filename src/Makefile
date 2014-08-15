# Makefile based *HEAVILY* on Jerry Cain's CS110 Makefiles

CXX = g++
RM = rm -f
OpenBLAS_DIR = /opt/OpenBLAS

# flags
#CXXFLAGS = -std=c++11 -march=native -Wall -O4 -I$(OpenBLAS_DIR)/include \
#						-L$(OpenBLAS_DIR)/lib
#LDFLAGS = -lopenblas -larmadillo

CXXFLAGS = -std=c++11 -march=native -Wall -O4 -I/usr/include \
	-I/usr/local/include -L/usr/local/lib -L/opt/intel/lib/intel64 \
	-L/opt/intel/mkl/lib/intel64

LDFLAGS = -larmadillo -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 \
	-lX11 -lpthread

# In this section, you list the files that are part of the project.
# If you add/change names of header/source files, here is where you
# edit the Makefile.
SOURCES = ar_model.cc sar.cc exact_sar.cc cheby_sar.cc car.cc \
					exact_car.cc cheby_car.cc ar_factory.cc
HEADERS = $(SOURCES:.cc=.h) create_w.h
OBJECTS = $(SOURCES:.cc=.o)

TEST = testing.out
all: test
test: $(TEST)

testing.out: $(OBJECTS) testing.cc
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# In make's default rules, a .o automatically depends on its .cc file
# (so editing the .cc will cause recompilation into its .o file).
# The line below creates additional dependencies, most notably that it
# will cause the .cc to rocompiled if any included .h file changes.

Makefile.dependencies:: $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -MM $(SOURCES) > Makefile.dependencies

-include Makefile.dependencies
# Phony means not a "real" target, it doesn't build anything
# The phony target "clean" is used to remove all compiled object files.
# The phony target "spartan" is used to remove all compiled object and backup files.
.PHONY: clean spartan all

clean:
	$(RM) $(CVX) $(CV) $(RP) $(OBJECTS) Makefile.dependencies

spartan: clean
	$(RM) *~

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
