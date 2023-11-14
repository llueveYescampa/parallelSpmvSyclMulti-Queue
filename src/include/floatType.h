#ifndef FLOATTYPE
  #ifdef DOUBLE
    //typedef double real;
    using floatType = double;
  #else
    //typedef float real;
    using floatType = float;
  #endif
  #define FLOATTYPE
#endif

