#include <stdlib.h>
#include <iostream>

#include "astra.hpp"

int main( int argc, char* argv[] ) {
  Kokkos::initialize( argc, argv );
  {
    Array3D<double>  toto("toto_name",16,16,16);
    Array3D<double>  totoprime("toto_prime",16,16,16);
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

    std::cout << "toto_host(1,1,1)=" << toto_host(1,1,1) << std::endl;

    std::cout << "j'ai fini!"<< std::endl;

  }
  Kokkos::finalize();
  return(0);
}


