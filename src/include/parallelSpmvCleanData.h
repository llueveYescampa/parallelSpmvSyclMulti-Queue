    delete[] w;    
/////////////// begin  de-allocating device memory //////////////////////////    
    free(w_d, myQueue[0]);
    free(v_d, myQueue[0]);
    free(vals_d, myQueue[0]);
    free(cols_d, myQueue[0]);
    free(rows_d, myQueue[0]);    
/////////////// end  de-allocating device memory //////////////////////////
    delete[] starRowQueue;
    
    delete[] ks;
    delete[] myQueue;


    delete[] v;
    delete[] val;
    delete[] col_idx;
    delete[] row_ptr;

