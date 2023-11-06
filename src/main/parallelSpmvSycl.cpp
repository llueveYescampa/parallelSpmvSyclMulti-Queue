#include <iostream>
#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;
using std::cout;
using std::ifstream;
using std::ios;

using ns = std::chrono::nanoseconds;

#include "parallelSpmvSycl.h"

#define MAXTHREADS 256
#define REP 1000


//  cout << "file: " << __FILE__  << " line: " << __LINE__ << "  nRowBlocks:" <<  nRowBlocks << '\n';

int main(int argc, char *argv[]) 
{
/*
    if (MAXTHREADS > 512) {
        printf("need to adjust the spmv() function to acomodate more than 512 threads per block\nQuitting ....\n");
        exit(-1);
    } // end if //
*/    
    
  #include "parallelSpmvData.h"

  auto meanAndSd = [] (floatType &mean, floatType &sd, const unsigned int *__restrict__ const data,  const unsigned int &n) -> void 
  {
    floatType sum = static_cast<floatType>(0);
    floatType standardDeviation = static_cast<floatType>(0);
    
    for(int row=0; row<n; ++row) {
        sum += (data[row+1] - data[row]);
    } // end for //
    mean = sum/n;
    
    for(int row=0; row<n; ++row) {
        standardDeviation += pow( (data[row+1] - data[row]) - mean, 2);
    } // end for //
    sd = sqrt(standardDeviation/n);
    return;
  }; // end of meanAndSd() lambda function;


  unsigned int n_global,nnz_global;

    // verifing number of input parameters //
  auto exists=true;
  auto checkSol=false;
    
    if (argc < 3 ) {
        cout << "Use: " <<  argv[0] <<   " Matrix_filename InputVector_filename  [SolutionVector_filename  [# of queues] ]\n";
        exit(-1);
    } // endif //
    
    ifstream inFile;
    // testing if matrix file exists
    inFile.open(argv[1], ios::in);
    if( !inFile ) {
        cout << "No matrix file found.\n";
        exists=false;
    } // end if //
    inFile.close();
    
    // testing if input file exists
    inFile.open(argv[2], ios::in);
    if( !inFile ) {
        cout << "No input vector file found.\n";
        exists=false;
    } // end if //
    inFile.close();

    // testing if output file exists
    if (argc  >3 ) {
        inFile.open(argv[3], ios::in);
        if( !inFile ) {
            cout << "No output vector file found.\n";
            exists=false;
        } else {
            checkSol=true;
        } // end if //
        inFile.close();        
    } // end if //

    if (!exists) {
        cout << "Quitting.....\n";
        exit(0);
    } // end if //
    
    
    // reading basic matrix data
    reader(n_global,nnz_global,&row_ptr,&col_idx,&val,argv[1]);
    // end of reading basic matrix data
    
    // finding tghe global nnzPerRow and stdDev
    meanAndSd(nnzPerRow, stdDev,row_ptr, n_global);

    v = new floatType[n_global];    
    vectorReader(v, n_global, argv[2]);

    if (checkSol) {  // put this down with the checks
      w = new floatType[n_global];
    } // end if //
    

////////////////// search for the raw number of block of rows   ////////////////////////////
/////////////////// determining the number of block rows based  ////////////////////////////
/////////////////// on global nnzPerRow and stdDev of the nnz   ////////////////////////////

  floatType temp = round(12.161 * log(stdDev/nnzPerRow) + 14.822);
  if (temp >1) nRowBlocks = temp;

  // the value of  nRowBlocks can be forced by run-time paramenter   
  if (argc  > 4  && atoi(argv[4]) > 0) {
      nRowBlocks = atoi(argv[4]);
  } // end if //  
  if (nRowBlocks > n_global) nRowBlocks = n_global;
  
  //cout << "file: " << __FILE__  << " line: " << __LINE__ << "  nRowBlocks:" <<  nRowBlocks << '\n';
/////////////// end of search for the raw number of block of rows   /////////////////////////

  //cout << ( (sizeof(floatType) == sizeof(double)) ? "Double": "Single" )  << " Precision. Solving dividing matrix into " << nRowBlocks << ((nRowBlocks > 1) ? " blocks": " block") << '\n';


////////////////// search for the real number of block of rows   ////////////////////////////
    unsigned int *blockSize = nullptr;
    unsigned int *starRowBlock = nullptr;
     
    blockSize = new unsigned int [nRowBlocks];
    starRowBlock = new unsigned int [nRowBlocks+1];
    
    for (int b=0; b<nRowBlocks; ++b) {
        blockSize[b] = 1;
    } // end for //

    starRowBlock[0]=0;
    getRowsNnzPerQueue(starRowBlock,n_global,nnz_global, row_ptr, nRowBlocks);
    
    for (int b=0; b<nRowBlocks; ++b) {
      
      meanAndSd(nnzPerRow, stdDev,&row_ptr[starRowBlock[b]], (starRowBlock[b+1]-starRowBlock[b]));
      /*
      cout << "\tnRows: " << (starRowBlock[b+1]-starRowBlock[b]) 
           << ", nnz: " << row_ptr[starRowBlock[b+1]] - row_ptr[starRowBlock[b]]
           <<  "\tinitial row of block: " << starRowBlock[b] +1
           <<  "\tfinal  row of block: " << starRowBlock[b+1] 
           <<   ", nnzPerRow: " << nnzPerRow 
           << ", stdDev: " << stdDev << '\n';
      */
      /////////////////////////////////////////////////////


      // these mean use vector spmv 
      floatType limit=nnzPerRow + parameter2Adjust*stdDev;
      //cout << "b: " << b << ", limit: " << limit << ", nnzPerRow: " << nnzPerRow << ", stdDev: " << stdDev <<  '\n';
      if ( limit < 4.5  ) {
          blockSize[b]=subGroupSize/32;
      }  else if (limit < 6.95 ) {
          blockSize[b]=subGroupSize/16;
      }  else if (limit < 15.5 ) {
          blockSize[b]=subGroupSize/8;
      }  else if (limit < 74.0 ) {
          blockSize[b]=subGroupSize/4;
      }  else if (limit < 300.0 ) {
          blockSize[b]=subGroupSize/2;
      }  else if (limit < 350.0 ) {
          blockSize[b]=subGroupSize;
      }  else if (limit < 1000.0 ) {
          blockSize[b]=subGroupSize*2;
      }  else if (limit < 2000.0 ) {
          blockSize[b]=subGroupSize*4;
      }  else if (limit < 3000.0 ) {
          blockSize[b]=subGroupSize*8;
      }  else {
          blockSize[b]=subGroupSize*16;
      } // end if //
      if (blockSize[b] > MAXTHREADS) {
          blockSize[b]=512;
      } // end if //    
      //printf("using vector spmv for on matrix,  blockSize: [%d, %d] %f, %f\n",blockSize[b],MAXTHREADS/blockSize[b], nnzPerRow, stdDev) ;
    } // end for //
    
    
    // here comes the consolidation ....
    nQueues=1;
    for (int b=1; b<nRowBlocks; ++b) {
        if (blockSize[b] != blockSize[b-1]) {
            ++nQueues;
        } // end if //
    } // end for //
    
   cout << "initial number of row sets: " << nRowBlocks 
        << ", final number of row sets: " << nQueues << '\n';

   
/////////////// end of search for the real number of block of rows //////////////////////////


/////////////// begin  creating queue dependent variables //////////////////////////

  //auto *q = new queue{property::queue::in_order()};
  myQueue = new queue[nQueues];
  ks  = new kernelDomain [nQueues];
/*
  /// listing device features //////
  cout << "Running on device: "          << myQueue[0].get_device().get_info<info::device::name>() << '\n';
  cout << "\tmax_compute_units: "        << myQueue[0].get_device().get_info<info::device::max_compute_units>() << '\n';
  cout << "\tmax_work_group_size: "      << myQueue[0].get_device().get_info<info::device::max_work_group_size>() << '\n';
  cout << "\tmax_work_item_dimensions: " << myQueue[0].get_device().get_info<info::device::max_work_item_dimensions>() << '\n';
  cout << "\tglobal_mem_size (Gb): "     << myQueue[0].get_device().get_info<info::device::global_mem_size>()/(1<<30) << '\n';
  cout << "\tlocal_mem_size (Kb): "      << myQueue[0].get_device().get_info<info::device::local_mem_size>()/(1<<10) << '\n';
  /// end of listing device features //////
*/

//  starRowStream = (int *) malloc( (nStreams+1) * sizeof(int) ); 
  starRowQueue = new unsigned int [nQueues+1];

    starRowQueue[0]       = starRowBlock[0];
    starRowQueue[nQueues] = starRowBlock[nRowBlocks];
    
    if (blockSize[0] > MAXTHREADS) {
        ks[0].workGroup  = range<2>{1,512};  
    } else {
        ks[0].workGroup  = range<2>{MAXTHREADS/blockSize[0],blockSize[0]};
    } // end if //    

    

  for (unsigned int b=1, q=1; b<nRowBlocks; ++b) {
  
    if (blockSize[b] != blockSize[b-1]) {
      if (blockSize[b] > MAXTHREADS) {
        ks[q].workGroup  = range<2>{1,512};  
      } else {
        ks[q].workGroup  = range<2>{MAXTHREADS/blockSize[b], blockSize[b]};  
      } // end if //    
      starRowQueue[q]=starRowBlock[b];
      ++q;
    } // end if //
  } // end for //



    for (unsigned int q=0; q<nQueues; ++q) {
        unsigned int nrows = starRowQueue[q+1]-starRowQueue[q];
        auto temp = ks[q].workGroup.get(0);
        ks[q].ndRange  = range<2>{temp*((nrows + temp - 1) / temp),ks[q].workGroup.get(1)};
        //ks[q].myKernel = nd_range<2> {ks[q].ndRange, ks[q].workGroup};
/*        
        cout << "\tblock for queue " << q 
             << " has size: [" 
             << ks[q].workGroup.get(0) 
             << ", " 
             << ks[q].workGroup.get(1) 
             << "],  and its grid has size: [" 
             << ks[q].ndRange.get(1) 
             << ", "
             << ks[q].ndRange.get(0) 
             << "], " 
             << starRowQueue[q+1]-starRowQueue[q] 
             << " rows and "
             << row_ptr[starRowQueue[q+1]] - row_ptr[starRowQueue[q]] 
             << " non-zeros.\n";             
*/             
    } // end for //
    delete[] starRowBlock;
    delete[] blockSize;
    
/////////////// end of creating queue dependent variables //////////////////////////



/////////////// begin  allocating device memory //////////////////////////
  rows_d = malloc_device<unsigned int>((n_global+1),myQueue[0]);
  cols_d = malloc_device<unsigned int>(nnz_global,myQueue[0]);
  vals_d = malloc_device<floatType>(nnz_global,myQueue[0]);
  v_d    = malloc_device<floatType>(n_global,myQueue[0]);
  w_d    = malloc_device<floatType>(n_global,myQueue[0]);
/////////////// end of allocating device memory //////////////////////////

  
/////////////// begin   copying memory to device  //////////////////////////

  myQueue[0].submit([rows_d, row_ptr,&n_global](handler &h) {
    // copy host_array to device_array
    h.memcpy(rows_d, row_ptr, (n_global+1) * sizeof(unsigned int));
  }); // end of submit() //

  myQueue[0].submit([cols_d, col_idx,&nnz_global](handler &h) {
    // copy host_array to device_array
    h.memcpy(cols_d, col_idx, nnz_global   * sizeof(unsigned int));
  }); // end of submit() //

  myQueue[0].submit([vals_d, val,&nnz_global](handler &h) {
    // copy host_array to device_array
    h.memcpy(vals_d, val    , nnz_global   * sizeof(floatType));
  }); // end of submit() //

  myQueue[0].submit([v_d, v,&n_global](handler &h) {
    // copy host_array to device_array
    h.memcpy(v_d   , v      , n_global    * sizeof(floatType));
  }); // end of submit() //
  
  myQueue[0].wait();
/////////////// end of  copying memory to device  //////////////////////////


/////////////// begin  testing spmv call //////////////////////////
  // Timing should begin here//

  auto start = std::chrono::steady_clock::now();

  for (int t=0; t<REP; ++t) {
    for (unsigned int q=0; q<nQueues; ++q) {
      const int sRow = starRowQueue[q];
      const int nrows = starRowQueue[q+1]-starRowQueue[q];    
      spmv((w_d+sRow), v_d,  vals_d, (rows_d+sRow), (cols_d), nrows, 1.0,0.0, myQueue[q],ks[q] );  
    } // end for //

    // is this neecsary?
    for (unsigned int q=0; q<nQueues; ++q) {
      myQueue[q].wait();
    } // end for //
  } // end for //      

/////////////// end of testing spmv call //////////////////////////

  auto duration = std::chrono::steady_clock::now() - start;
  auto elapsed_time = std::chrono::duration_cast<ns>(duration).count();
  
  
  cout << "Total time was " << elapsed_time*1.0e-9 
       << " seconds, GFLOPS: " << (2.0*nnz_global+ 3.0*n_global)*REP/elapsed_time
       << ", GBytes/s: " << (nnz_global*(2*sizeof(floatType) + sizeof(int))+n_global*(sizeof(floatType)+sizeof(int)))*REP*1.0/elapsed_time << '\n';



/////////////// begin  copying memory to device  //////////////////////////

  myQueue[0].submit([w,w_d,&n_global](handler &h) {
    // copy host_array to device_array
    h.memcpy(w   , w_d      , n_global    * sizeof(floatType));
  }); // end of submit() //
  myQueue[0].wait();
/////////////// end of copying memory to device  //////////////////////////





/*    
        
  
    // Timing should begin here//

    struct timeval tp;                                   // timer
    double elapsed_time;
    gettimeofday(&tp,NULL);  // Unix timer
    elapsed_time = -(tp.tv_sec*1.0e6 + tp.tv_usec);
    for (int t=0; t<REP; ++t) {

        //cuda_ret = cudaMemset(w_d, 0, (size_t) n_global*sizeof(floatType) );
        //if(cuda_ret != cudaSuccess) FATAL("Unable to set device for matrix w_d");
        
        for (int s=0; s<nStreams; ++s) {
            const int sRow = starRowStream[s];
            const int nrows = starRowStream[s+1]-starRowStream[s];
#ifdef USE_TEXTURE
            spmv<<<grid[s], block[s], sharedMemorySize[s], stream[s] >>>((w_d+sRow), v_t, vals_d, (rows_d+sRow), (cols_d), nrows, 1.0,0.0);
#else
            spmv<<<grid[s], block[s], sharedMemorySize[s], stream[s] >>>((w_d+sRow), v_d,  vals_d, (rows_d+sRow), (cols_d), nrows, 1.0,0.0);
#endif
        } // end for //
        
        for (int s=0; s<nStreams; ++s) {
            //cudaStreamSynchronize(NULL);
            cudaStreamSynchronize(stream[s]);
        } // end for //
        
    } // end for //    
    
    
    gettimeofday(&tp,NULL);
    elapsed_time += (tp.tv_sec*1.0e6 + tp.tv_usec);
    printf ("Total time was %f seconds, GFLOPS: %f, GBytes/s: %f\n", elapsed_time*1.0e-6, 
                                (2.0*nnz_global+ 3.0*n_global)*REP*1.0e-3/elapsed_time,
                                (nnz_global*(2*sizeof(floatType) + sizeof(int))+n_global*(sizeof(floatType)+sizeof(int)))*REP*1.0e-3/elapsed_time );
    

    cuda_ret = cudaMemcpy(w, w_d, (n_global)*sizeof(floatType),cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix y_d back to host");

// cuda stuff ends here
//////////////////////////////////////

*/    
   
    if (checkSol) {
    
        floatType *sol=nullptr;
        sol = new floatType[n_global];
        // reading input vector
        vectorReader(sol, n_global, argv[3]);
        
        int row=0;
        floatType tolerance = 1.0e-08;
        if (sizeof(floatType) != sizeof(double) ) {
            tolerance = 1.0e-02;
        } // end if //

        floatType error;
        do {
            error =  fabs(sol[row] - w[row]) /fabs(sol[row]);
            if ( error > tolerance ) break;
            ++row;
        } while (row < n_global); // end do-while //
        
        if (row == n_global) {
            printf("Solution match in GPU\n");
        } else {    
            printf("For Matrix %s, solution does not match at element %d in GPU  %20.13e   -->  %20.13e  error -> %20.13e, tolerance: %20.13e \n", 
            argv[1], (row+1), sol[row], w[row], error , tolerance  );
        } // end if //
        delete[] sol;
        
    } // end if //
    //free(w);
    //free(v);
    
//#ifdef USE_TEXTURE
//    cudaDestroyTextureObject(v_t);
//#endif    

    #include "parallelSpmvCleanData.h" 
    return 0;    
} // end main() //


/*
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
*/


/*
void meanAndSd(floatType *mean, floatType *sd,floatType *data, int n)
{
    floatType sum = (floatType) 0.0; 
    floatType standardDeviation = (floatType) 0.0;

    for(int i=0; i<n; ++i) {
        sum += data[i];
    } // end for //

    *mean = sum/n;

    for(int i=0; i<n; ++i) {
        standardDeviation += pow(data[i] - *mean, 2);
    } // end for //
    *sd=sqrt(standardDeviation/n);
} // end of meanAndSd //
*/


/*
    // ready to start //    
    cudaError_t cuda_ret;
    
//////////////////////////////////////
// cuda stuff start here

    // Allocating device memory for input matrices 

    cuda_ret = cudaMalloc((void **) &rows_d,  (n_global+1)*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for rows_d");
    
    cuda_ret = cudaMalloc((void **) &cols_d,  (nnz_global)*sizeof(int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for cols_d");
    
    cuda_ret = cudaMalloc((void **) &vals_d,  (nnz_global)*sizeof(floatType));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for vals_d");

    cuda_ret = cudaMalloc((void **) &v_d,  (n_global)*sizeof(floatType));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for x_d");

    cuda_ret = cudaMalloc((void **) &w_d,  (n_global)*sizeof(floatType));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for y_d");

   // Copy the input matrices from the host memory to the device memory

    cuda_ret = cudaMemcpy(rows_d, row_ptr, (n_global+1)*sizeof(int),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix rows_d");

    cuda_ret = cudaMemcpy(cols_d, col_idx, (nnz_global)*sizeof(int),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix cols_d");

    cuda_ret = cudaMemcpy(vals_d, val, (nnz_global)*sizeof(floatType),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix vals_d");


    cuda_ret = cudaMemcpy(v_d, v, (n_global)*sizeof(floatType),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device matrix x_d");

#ifdef USE_TEXTURE

    cudaTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = cudaAddressModeClamp;
    td.readMode = cudaReadModeElementType;


    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = v_d;
    resDesc.res.linear.sizeInBytes = n_global*sizeof(floatType);
    #ifdef DOUBLE
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.y = 32;
    #else
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    #endif
    resDesc.res.linear.desc.x = 32;

    cudaTextureObject_t v_t;
    cuda_ret = cudaCreateTextureObject(&v_t, &resDesc, &td, NULL);
    if(cuda_ret != cudaSuccess) FATAL("Unable to create text memory v_t");
    

    //cuda_ret = cudaBindTexture(NULL, v_t, v_d, n_global*sizeof(floatType));
    //cuda_ret = cudaBindTexture(NULL, valTex, vals_d, nnz_global*sizeof(floatType));            
  
  
#endif
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);



////////////////// search for the  number of block of rows  ///////////////////////
    { // determining the number of block rows based on mean and sd of the nnz 
        // opening matrix file to read mean and sd of number of nonzeros per row
        double tmpMean, tmpSD;
        fh = fopen(argv[1], "rb");
        // reading laast two values in file: mean and sd //
        fseek(fh, 0L, SEEK_END);
        long int offset = ftell(fh)-2*sizeof(double);
        fseek(fh, offset, SEEK_SET);
        if ( !fread(&tmpMean, sizeof(double), (size_t) 1, fh)) exit(0);
        if ( !fread(&tmpSD, sizeof(double), (size_t) 1, fh)) exit(0);
        if (fh) fclose(fh);
        
        // determining number of streams based on mean and sd
        floatType ratio = tmpSD/tmpMean;
        //printf("file: %s, line: %d, tMean nnz: %.2f, SD nnz: %.2f, ratio: %.2f\n", __FILE__, __LINE__ , tmpMean, tmpSD, ratio);
        nRowBlocks = round(12.161 * log(ratio) + 14.822);
        if (nRowBlocks < 1 ) nRowBlocks=1;               
        printf("nRowBlocks: %d\n", nRowBlocks);
    } // end of determining the number of block rows based on mean and sd of the nnz
    
     
    // the value of  nRowBlocks can be forced by run-time paramenter   
    if (argc  > 4  && atoi(argv[4]) > 0) {
        nRowBlocks = atoi(argv[4]);
    } // end if //
    if (nRowBlocks > n_global) nRowBlocks = n_global;
////////////////// end of search for the  number of block of rows  /////////////////////////
*/

