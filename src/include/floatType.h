#ifndef FLOATTYPE
  #include <sycl/sycl.hpp>
  using namespace sycl;
  typedef struct kernelDomain {
    range<2> ndRange{1,1};
    range<2> workGroup{1,1};
    //nd_range<2> myKernel{ndRange, workGroup};
  } kernelDomain;


  #ifdef DOUBLE
    //typedef double real;
    using floatType = double;
  #else
    //typedef float real;
    using floatType = float;
  #endif
  #define FLOATTYPE
#endif

