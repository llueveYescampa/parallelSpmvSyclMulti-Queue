    // begining of data for solution

    unsigned int n_global,nnz_global;
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

    // end of data for solution
