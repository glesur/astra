#include <stdlib.h>
#include <iostream>

#include "astra.hpp"
#include "loop.hpp"
#include "global.hpp"
#include "input.hpp"
#include "grid.hpp"
#include "field.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>

int main( int argc, char* argv[] ) {
  Kokkos::initialize( argc, argv );
  {
    // Initialize astra screen outputs, logs, profiling
    astra::initialize();

    ///////////////////////////////////
    //Initialisation phase
    // Read input
    Input input(argc, argv);
    input.PrintLogo();

    Grid grid(input);
    
    // Show configuration after initialisation
    input.ShowConfig();
    grid.ShowConfig();
    /////////////////////////////////////////
    // Test of 1D FFT
    const int n = 4;

    Field<Array3D<real>> fld("myField",grid.npr);
    fld.Add("vx");
    // Test a derivative
    auto vxi = fld["vx"];
    auto xi = grid.x[IDIR];
    astra_for("loop_example",fld,
      KOKKOS_LAMBDA(int i,int j,int k) {
        vxi(i,j,k) = std::sin(2.0*M_PI*xi(i));
    }); 
    astra::cout << "Initial condition : " << std::endl;
    for(int i=0 ; i < grid.npr[IDIR] ; i++) {
      astra::cout << vxi(i,0,0) << " ; ";
    }
    astra::cout << std::endl;

    Field<Array3D<real>> fldo("myField",grid.npr);
    fldo.Add("vx");

    Field<Array3D<complex>> fldf("myField",grid.npf);
    fldf.Add("vx");

    // test 3D FFT
    Kokkos::fence();


    KokkosFFT::rfftn(Kokkos::DefaultExecutionSpace(), fld["vx"], fldf["vx"]);
    auto vxf = fldf["vx"];
    auto kx1 = grid.kx[IDIR];
    astra_for("loop_example",fldf,
      KOKKOS_LAMBDA(int i,int j,int k) {
        vxf(i,j,k) *= kx1(i)*Kokkos::complex(0.0, 1.0);
    }); 

    KokkosFFT::irfftn(Kokkos::DefaultExecutionSpace(), fldf["vx"], fldo["vx"]);

    auto vxo = fldo["vx"];

    astra::cout << "After derivative : " << std::endl;
    for(int i=0 ; i < grid.npr[IDIR] ; i++) {
      astra::cout << vxo(i,0,0) << " ; ";
    }
    astra::cout << std::endl;

    // Test FFTs
    Array1D<real> x("x", n);
    Array1D<complex> x_hat("x_hat", n/2+1);

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    Kokkos::fill_random(x, random_pool, 1);
    Kokkos::fence();

    KokkosFFT::rfft(Kokkos::DefaultExecutionSpace(), x, x_hat);

    // Test standard kokkos array
    Array3D<real>  toto("toto_name",16,16,16);
    Array3D<real>  totoprime("toto_prime",16,16,16);
    astra_for("loop_example",0,9,0,9,0,9,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        toto(i,j,k) = 1.0;
      });

    astra_for("loop_example2",0,9,0,9,0,9,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        totoprime(i,j,k) = toto(i,j,k)+toto(i,j,k);
      });

      astra_for("loop_example3",0,9,0,9,0,9,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        toto(i,j,k) = toto(i,j,k) + totoprime(i,j,k);
      });
    
    ArrayHost3D<double> toto_host("toto_host",16,16,16);
    Kokkos::deep_copy(toto_host,toto);

    astra::cout << "toto_host(1,1,1)=" << toto_host(1,1,1) << std::endl;

    astra::cout << "j'ai fini!"<< std::endl;

  }
  Kokkos::finalize();
  return(0);
}


