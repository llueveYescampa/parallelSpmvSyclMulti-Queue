#include <sycl/sycl.hpp>
using namespace sycl;

#include "parallelSpmvSycl.h"

void spmv(       floatType     *__restrict const y, 
           const floatType     *__restrict const x, 
           const floatType     *__restrict const val, 
           const unsigned int  *__restrict const row_ptr, 
           const unsigned int  *__restrict const col_idx, 
           const unsigned int                   &nRows,
           const floatType                      &alpha,
           const floatType                      &beta,
                 queue                          &q,
           const kernelDomain                   &kd
          )
{   
  q.submit([ & ](handler &h) {
    auto temp = local_accessor<floatType, 1>(range<1>{kd.workGroup[0]*kd.workGroup[1]}, h);

    h.parallel_for(nd_range<2>{kd.ndRange,kd.workGroup}, [=](const nd_item<2> &nd_item) [[sycl::reqd_sub_group_size(32)]] {
      //const auto sg = nd_item.get_sub_group();
      //const auto warpSize = sg.get_max_local_range()[0];
      //const auto blockRows = nd_item.get_local_range(0);
      const auto blockCols = nd_item.get_local_range(1);
    
      const auto row = nd_item.get_global_id(0);
      //const auto col = nd_item.get_global_id(1);
      
      //const auto localRow = nd_item.get_local_id(0);
      const auto localCol = nd_item.get_local_id(1);
      const auto index = nd_item.get_local_id(0) * nd_item.get_local_range(1) + nd_item.get_local_id(1);  
      auto sum = static_cast<floatType>(0);
        
      if (row < nRows) {      
          for (unsigned int col=row_ptr[row]+localCol; col < row_ptr[row+1]; col+= blockCols  ) {
              sum += (val[col] * x[col_idx[col]]);
              //temp[index] += (val[col] * x[col_idx[col]]);
          } // end for //
      } // end if //
      temp[index]=sum;
        
      switch(blockCols) {
        case 1024 :
          group_barrier(nd_item.get_group());
          if (localCol < 512) temp[index] += temp[index + 512]; 
        case 512 :
          group_barrier(nd_item.get_group());
          if (localCol < 256) temp[index] += temp[index + 256]; 
        case 256 :      
          group_barrier(nd_item.get_group());
          if (localCol < 128) temp[index] += temp[index + 128]; 
        case 128 :
          group_barrier(nd_item.get_group());
          if (localCol < 64)  temp[index] += temp[index +  64];
        case  64 :
          group_barrier(nd_item.get_group());
          if (localCol < 32) temp[index] += temp[index +  32];  
        case  32 :
          if (localCol < 16) temp[index] += temp[index +  16];  
        case  16 :
          if (localCol <  8) temp[index] += temp[index +   8];  
        case   8 :
          if (localCol <  4) temp[index] += temp[index +   4];  
        case   4 :
          if (localCol <  2) temp[index] += temp[index +   2];  
        case   2 :
          if (localCol <  1) temp[index] += temp[index +   1];  
      } // end switch //
        
      if (row < nRows and localCol == 0) {
        //y[row] += temp[sharedMemIndx];
        y[row] =  beta * y[row] + alpha*temp[index];
      } // end if //          
    }); // end of parallel_for() //
  }); // end of submit() //
} // end of spmv() //