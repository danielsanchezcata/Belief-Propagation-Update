CPP=g++-14
CPPFLAGS=-std=c++17 -march=native -I ./inc -I /usr/local/include -I /opt/homebrew/include/eigen3 -I /usr/local/Cellar/eigen/3.4.0_1/include/eigen3 -I /opt/homebrew/include -O3 -fopenmp
LDFLAGS=-L /opt/homebrew/lib -lfftw3 -lfftw3l -lquadmath -lm

all: LEMP pop-dyn-planted fixed-cost arb-kernel

$(info Using compiler: $(CPP))

LEMP:
	$(CPP) $(CPPFLAGS) -c -fPIC src/LEMP.cpp -o out/LEMP.o
	$(CPP) $(CPPFLAGS) $(LDFLAGS) -shared -Wl,-install_name,LEMP.so -o out/LEMP.so out/LEMP.o

pop-dyn-planted:
	$(CPP) $(CPPFLAGS) -c -fPIC src/PopDynPlanted.cpp -o out/PDP.o
	$(CPP) $(CPPFLAGS) $(LDFLAGS) -shared -Wl,-install_name,PDP.so -o out/PDP.so out/PDP.o

fixed-cost:
	$(CPP) $(CPPFLAGS) -c -fPIC src/fixed-cost.cpp -o out/fixedCost.o
	$(CPP) $(CPPFLAGS) $(LDFLAGS) -shared -Wl,-install_name,fixedCost.so -o out/fixedCost.so out/fixedCost.o

arb-kernel:
	$(CPP) $(CPPFLAGS) -c -fPIC src/arbitrary-kernel.cpp -o out/arbKernel.o
	$(CPP) $(CPPFLAGS) $(LDFLAGS) -shared -Wl,-install_name,arbKernel.so -o out/arbKernel.so out/arbKernel.o
