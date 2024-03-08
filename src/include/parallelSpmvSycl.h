#ifndef PARALLELSPMV_H
#define PARALLELSPMV_H
#include "floatType.h"

    #include <sycl/sycl.hpp>
    using namespace sycl;
    typedef struct kernelDomain {
      range<2> ndRange{1,1};
      range<2> workGroup{1,1};
      //nd_range<2> myKernel{ndRange, workGroup};
    } kernelDomain;



    void reader(unsigned int &gn, 
                unsigned int &gnnz, 
                unsigned int **rPtr,
                unsigned int **cIdx,
                floatType **v,
                const char *const matrixFile);


    void vectorReader(floatType *const v, unsigned int &n, const char *const vectorFile);


    void getRowsNnzPerQueue(      unsigned int *__restrict const rowsPerSet, 
                            const unsigned int &global_n, 
                            const unsigned int &global_nnz,  
                            const unsigned int *__restrict const rows, 
                            const unsigned int &nRowBlocks);
        


    
              
    void spmv(       floatType     *__restrict const y, 
               const floatType     *__restrict const x, 
               const floatType     *__restrict const val, 
               const unsigned int  *__restrict const row_ptr, 
               const unsigned int  *__restrict const col_idx, 
               const unsigned int                      &nRows,
               const floatType                         &alpha,
               const floatType                         &beta,
                     queue                             &q,
               const kernelDomain                      &kd
              );
#endif
