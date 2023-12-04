#include <sycl/sycl.hpp>
using namespace sycl;

#include "parallelSpmvSycl.h"

template <const unsigned int bs>
void subgroupReduce(volatile floatType *__restrict__  const temp1) {
  // unrolling warp 
  if (bs >= 64) temp1[0] += temp1[32];
  if (bs >= 32) temp1[0] += temp1[16];
  if (bs >= 16) temp1[0] += temp1[ 8];
  if (bs >=  8) temp1[0] += temp1[ 4];
  if (bs >=  4) temp1[0] += temp1[ 2];
  if (bs >=  2) temp1[0] += temp1[ 1];
} // end of subgroupReduce() //

void spmv(       floatType *__restrict__ const y, 
           const floatType *__restrict__ const x, 
           const floatType *__restrict__ const val, 
           const unsigned int  *__restrict__ const row_ptr, 
           const unsigned int  *__restrict__ const col_idx, 
           const unsigned int nRows,
           const floatType &alpha,
           const floatType &beta,
                 queue &q,
           const kernelDomain &kd           
          )
{   

  q.submit([ & ](handler &h) {
    auto temp = local_accessor<floatType, 2>(kd.workGroup, h);

    h.parallel_for(nd_range<2>{kd.ndRange,kd.workGroup}, [=](const nd_item<2> &nd_item) {
      //const auto sg = nd_item.get_sub_group();
      //const auto warpSize = sg.get_max_local_range()[0];
      //const auto blockRows = nd_item.get_local_range(0);
      const auto blockCols = nd_item.get_local_range(1);

      const auto row = nd_item.get_global_id(0);
      //const auto col = nd_item.get_global_id(1);

      const auto localRow = nd_item.get_local_id(0);
      const auto localCol = nd_item.get_local_id(1);
      temp[localRow][localCol] =  static_cast<floatType>(0);
            
      if (row < nRows) {    
        switch(blockCols) {
        case   1 :
          for (unsigned int col=row_ptr[row]+localCol; col < row_ptr[row+1]; col+= blockCols  ) {
            temp[localRow][localCol] += (val[col] * x[col_idx[col]]);
          } // end for //

          if (localCol  == 0) {
            //y[row] += temp[sharedMemIndx];
            y[row] =  beta * y[row] + alpha*temp[localRow][localCol];
          } // end if //
          break;
          case   2 :
            for (unsigned int col=row_ptr[row]+localCol; col < row_ptr[row+1]; col+= blockCols  ) {
              temp[localRow][localCol] += (val[col] * x[col_idx[col]]);
            } // end for //

            if (localCol <  1) subgroupReduce<2>(&temp[localRow][localCol]);

            if (localCol  == 0) {
              //y[row] += temp[sharedMemIndx];
              y[row] =  beta * y[row] + alpha*temp[localRow][localCol];
            } // end if //
          break;          
        case   4 :
          for (unsigned int col=row_ptr[row]+localCol; col < row_ptr[row+1]; col+= blockCols  ) {
            temp[localRow][localCol] += (val[col] * x[col_idx[col]]);
          } // end for //

          if (localCol <  2) subgroupReduce<4>(&temp[localRow][localCol]);

          if (localCol  == 0) {
            //y[row] += temp[sharedMemIndx];
            y[row] =  beta * y[row] + alpha*temp[localRow][localCol];
          } // end if //
          break;
        case   8 :
          for (unsigned int col=row_ptr[row]+localCol; col < row_ptr[row+1]; col+= blockCols  ) {
            temp[localRow][localCol] += (val[col] * x[col_idx[col]]);
          } // end for //

          if (localCol <  4) subgroupReduce<8>(&temp[localRow][localCol]);

          if (localCol  == 0) {
            //y[row] += temp[sharedMemIndx];
            y[row] =  beta * y[row] + alpha*temp[localRow][localCol];
          } // end if //
          break;
        case  16 :
          for (unsigned int col=row_ptr[row]+localCol; col < row_ptr[row+1]; col+= blockCols  ) {
            temp[localRow][localCol] += (val[col] * x[col_idx[col]]);
          } // end for //

          if (localCol <  8) subgroupReduce<16>(&temp[localRow][localCol]);

          if (localCol  == 0) {
            //y[row] += temp[sharedMemIndx];
            y[row] =  beta * y[row] + alpha*temp[localRow][localCol];
          } // end if //
          break;
        case  32 :
          for (unsigned int col=row_ptr[row]+localCol; col < row_ptr[row+1]; col+= blockCols  ) {
            temp[localRow][localCol] += (val[col] * x[col_idx[col]]);
          } // end for //

          if (localCol < 16) subgroupReduce<32>(&temp[localRow][localCol]);

          if (localCol  == 0) {
            //y[row] += temp[sharedMemIndx];
            y[row] =  beta * y[row] + alpha*temp[localRow][localCol];
          } // end if //
          break;
        case  64 :
          for (unsigned int col=row_ptr[row]+localCol; col < row_ptr[row+1]; col+= blockCols  ) {
            temp[localRow][localCol] += (val[col] * x[col_idx[col]]);
          } // end for //

          group_barrier(nd_item.get_group());

          if (localCol < 32) subgroupReduce<64>(&temp[localRow][localCol]);

          if (localCol  == 0) {
            //y[row] += temp[sharedMemIndx];
            y[row] =  beta * y[row] + alpha*temp[localRow][localCol];
          } // end if //
          break;
        case 128 :
          for (unsigned int col=row_ptr[row]+localCol; col < row_ptr[row+1]; col+= blockCols  ) {
            temp[localRow][localCol] += (val[col] * x[col_idx[col]]);
          } // end for //

          group_barrier(nd_item.get_group());              
          if (localCol < 64)  temp[localRow][localCol] += temp[localRow][localCol+ 64];

          group_barrier(nd_item.get_group());               
          if (localCol < 32 ) subgroupReduce<64>(&temp[localRow][localCol]);

          if (localCol  == 0) {
            //y[row] += temp[sharedMemIndx];
            y[row] =  beta * y[row] + alpha*temp[localRow][localCol];
          } // end if //
          break;
        case 256 :      
          for (unsigned int col=row_ptr[row]+localCol; col < row_ptr[row+1]; col+= blockCols  ) {
            temp[localRow][localCol] += (val[col] * x[col_idx[col]]);
          } // end for //
          group_barrier(nd_item.get_group());
          if (localCol < 128) temp[localRow][localCol] += temp[localRow][localCol+128]; 

          group_barrier(nd_item.get_group());             
          if (localCol < 64)  temp[localRow][localCol] += temp[localRow][localCol+ 64];

          group_barrier(nd_item.get_group());               
          if (localCol < 32 ) subgroupReduce<64>(&temp[localRow][localCol]);
          if (localCol  == 0) {
            //y[row] += temp[sharedMemIndx];
            y[row] =  beta * y[row] + alpha*temp[localRow][localCol];
          } // end if //
          break;
        case 512 :
          for (unsigned int col=row_ptr[row]+localCol; col < row_ptr[row+1]; col+= blockCols  ) {
            temp[localRow][localCol] += (val[col] * x[col_idx[col]]);
          } // end for //
          group_barrier(nd_item.get_group());            
          if (localCol < 256) temp[localRow][localCol] += temp[localRow][localCol+256]; 

          group_barrier(nd_item.get_group());             
          if (localCol < 128) temp[localRow][localCol] += temp[localRow][localCol+128]; 

          group_barrier(nd_item.get_group());             
          if (localCol < 64)  temp[localRow][localCol] += temp[localRow][localCol+ 64];

          group_barrier(nd_item.get_group()); 
          if (localCol < 32 ) subgroupReduce<64>(&temp[localRow][localCol]);
          if (localCol  == 0) {
            //y[row] += temp[sharedMemIndx];
            y[row] =  beta * y[row] + alpha*temp[localRow][localCol];
          } // end if //
          break;
        } // end switch //
      } // end if //  
    }); // end of parallel_for() //
  }); // end of submit() //
} // end of spmv() //

