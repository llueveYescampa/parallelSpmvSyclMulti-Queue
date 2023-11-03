//# define DEFAULT_STREAMS 4
//# define MAX_STREAMS 16

    unsigned int *row_ptr = nullptr;
    unsigned int *col_idx = nullptr;
    floatType *val = nullptr;
    floatType *w = nullptr;
    floatType *v = nullptr;
    unsigned int nRowBlocks=1;
    floatType nnzPerRow, stdDev;

    const unsigned int subGroupSize = 32;
    const floatType parameter2Adjust = 0.15;
    unsigned int nQueues;

    queue *myQueue = nullptr;
    kernelDomain *ks = nullptr;
    unsigned int *starRowQueue = nullptr;

    unsigned int *rows_d = nullptr;
    unsigned int *cols_d = nullptr;
    floatType *vals_d = nullptr;
    floatType *v_d = nullptr;
    floatType *w_d = nullptr;

 
/*
    floatType meanNnzPerRow=static_cast<floatType>(0);




    // data for the on_proc solution

    int *rows_d, *cols_d;
    floatType *vals_d;
    floatType *v_d, *w_d;

    // end of data for the on_proc solution
    
    
    
    
    floatType sd=0;
    
    dim3 *block=NULL;
    dim3 *grid=NULL;

    size_t *sharedMemorySize=NULL;
*/
