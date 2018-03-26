
/*
 Note from Khan:
 This program is currently working correctly with 'scale'.
 Contains 2 mkl-blas calls and 2 cublas calls for Sgemm
 Extra printf stmts have been removed
 Debug code has been added
 Usage: ./simpleCUBLAS threads size [scale] [debug]
*/

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

/* Includes, MKL */
#include <mkl.h>
#include <omp.h>

/* Matrix size */
//#define N  (4)
#define min(x,y) (((x) > (y)) ? (x) : (y))
#define MAX(a,b) (((a)>(b))?(a):(b))

/* Main */
int main(int argc, char **argv)
{
    cublasStatus_t status;
    float *h_A;
    float *h_B;
    float *cublas_C;
    float *mkl_C;
//    float *d_A = 0;
//    float *d_B = 0;
//    float *d_C = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    //int n2 = N * N;
    int i;
    float error_norm;
    float ref_norm;
    float diff;
//    cublasHandle_t handle;

    /* Extra code added to take args from cmd line */

    int debug = 0;
    float scale = 0;
    double starttime, stoptime,starttime2, stoptime2,starttime_ser,stoptime_ser;
    double stage1stop,stage2stop,stage1start,stage2start,starttime_vect,stoptime_vect;
    int mythreads,nthreads,tid;;
    int m,n,p,N;
    int detailedtime = 1;
    // A = m x p ;  B = p x n ; C = m x n

    mythreads = atoi(argv[1]);
    N = atoi(argv[2]);
    if (argc > 3)
        scale = atof(argv[3]);
    if (argc > 4)
        debug = atoi(argv[4]);
    printf("Taking scale as : %f , SIZE as : %d\n",scale,N);
    int n2 = N * N;
    m = n = p = N;
    float var1, var2;

    var1 = scale*N;
    var2 = (1-scale)*N;

   if (!(fmod(var1, 1.0) == 0.0 && fmod(var2, 1.0) == 0.0))
   {
       printf("Mul1 = |%f| , Mul2 = |%f| is NOT int\n",var1, var2 );
       printf("Exiting\n");
       exit(0);
   }
    /* Extra code ends here */

//    int dev = findCudaDevice(argc, (const char **) argv);
//
//    if (dev == -1)
//    {
//        return EXIT_FAILURE;
//    }

    /* Initialize CUBLAS */
   // printf("simpleCUBLAS test running..\n");

//    status = cublasCreate(&handle);
//
//    if (status != CUBLAS_STATUS_SUCCESS)
//    {
//        fprintf(stderr, "!!!! CUBLAS initialization error\n");
//        return EXIT_FAILURE;
//    }

    /* Allocate host memory for the matrices */
    h_A = (float *)malloc(n2 * sizeof(h_A[0]));

    if (h_A == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }

    h_B = (float *)malloc(n2 * sizeof(h_B[0]));

    if (h_B == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }

    mkl_C = (float *)malloc(n2 * sizeof(mkl_C[0]));

    if (mkl_C == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (mkl_C)\n");
        return EXIT_FAILURE;
    }
    cublas_C = (float *)malloc(n2 * sizeof(cublas_C[0]));

    if (cublas_C == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Fill the matrices with test data */
    for (i = 0; i < n2; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        // cublas_C[i] = rand() / (float)RAND_MAX;
        cublas_C[i] = 0.0;
        mkl_C[i] = 0.0;
//           h_A[i] = (rand() %10)+1;
//           h_B[i] = (rand() %10)+1;
        //   cublas_C[i] = rand() % 10;

    }




    /* Performs operation using plain C code */

    // Setup row major format for CUBLAS
    float* AT = h_A;
    float* BT = h_B;
    //float* CT = cublas_C;
    int k;
    int num_row_A,num_col_A, num_row_AT,num_col_AT;
    int num_row_B,num_col_B, num_row_BT,num_col_BT;
    int num_row_C,num_col_C, num_row_CT,num_col_CT;

    int lda = num_col_A = num_row_AT = N;
    int ldb = num_col_B = num_row_BT = N;
    int ldc = num_row_C = N;
    m = num_row_C = num_row_AT = num_col_A = N;
    n = num_col_C = num_row_BT = num_col_B = N;
    k = num_col_AT = num_row_B = N;

    mkl_set_dynamic(0);
    //printf("The number of threads before %d \n", mkl_get_max_threads());
    mkl_set_num_threads(mythreads);
    //printf("The number of threads after %d \n", mkl_get_max_threads());
    omp_set_nested(1);

    //for ref
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, p, alpha, h_A, p, h_B, n, beta, mkl_C, n);
    //

    // Stage 1 - 1st half cublas, 2nd half mkl
    // This stage stores the result matrix in cublas_C

    printf("Stage 1\n");
    stage1start = omp_get_wtime();
    #pragma omp parallel sections private(nthreads, tid) num_threads(2)
    {
    	 /* Obtain thread number */
	    tid = omp_get_thread_num();
	   //  printf("Hello World from thread = %d\n", tid);

	    // /* Only master thread does this */
	    if (tid == 0)
	    {
	      nthreads = omp_get_num_threads();
	    }
        #pragma omp section
	    {
	        float *d_A = 0;
            float *d_B = 0;
            float *d_C = 0;
            cublasHandle_t handle;

            tid = omp_get_thread_num();
//            printf(" Sec 1a\n");

            nthreads = omp_get_num_threads();

            checkCudaErrors(cudaSetDevice(0));
            //cublasInit();
//            printf("\n  1 Dev Number : %d\n",0);

             status = cublasCreate(&handle);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "!!!! CUBLAS initialization error\n");
                //return EXIT_FAILURE;
            }


                     /* Allocate device memory for the matrices */
            if (cudaMalloc((void **)&d_A, n2 * sizeof(d_A[0])) != cudaSuccess)
            {
                fprintf(stderr, "!!!! 1 device memory allocation error (allocate A)\n");
                //return EXIT_FAILURE;
            }

            if (cudaMalloc((void **)&d_B, n2 * sizeof(d_B[0])) != cudaSuccess)
            {
                fprintf(stderr, "!!!! 1 device memory allocation error (allocate B)\n");
               // return EXIT_FAILURE;
            }

            if (cudaMalloc((void **)&d_C, n2 * sizeof(d_C[0])) != cudaSuccess)
            {
                fprintf(stderr, "!!!! 1 device memory allocation error (allocate C)\n");
                //return EXIT_FAILURE;
            }

            if (detailedtime)
            {
            starttime2 = omp_get_wtime();

            }


            /* Initialize the device matrices with the host matrices */

            starttime_vect = omp_get_wtime();

            status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "!!!! 1 device access error (write B)\n");
                //return EXIT_FAILURE;
            }

           // status = cublasSetVector(n2, sizeof(cublas_C[0]), cublas_C, 1, d_C, 1);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "!!!! 1 device access error (write C)\n");
               // return EXIT_FAILURE;
            }
            stoptime_vect = omp_get_wtime();

            //for 1st half of Cublas
            status = cublasSetVector(n2*scale, sizeof(h_A[0]), h_A, 1, d_A, 1);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
            fprintf(stderr, "!!!! 1 device access error (1 write A)\n");
            //return EXIT_FAILURE;
            }
            status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m*(scale), k, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
            fprintf(stderr, "!!!! 1 kernel execution error.\n");
            //return EXIT_FAILURE;
            }
            /* Read the result back */
            status = cublasGetVector(n2*scale, sizeof(cublas_C[0]), d_C, 1, cublas_C, 1);


            if (status != CUBLAS_STATUS_SUCCESS)
            {
            fprintf(stderr, "!!!! 1 device access error (1 read C)\n");
            //return EXIT_FAILURE;
            }


            if (cudaFree(d_A) != cudaSuccess)
            {
                fprintf(stderr, "!!!! 1 memory free error (A)\n");
                //return EXIT_FAILURE;
            }

            if (cudaFree(d_B) != cudaSuccess)
            {
                fprintf(stderr, "!!!! 1 memory free error (B)\n");
                //return EXIT_FAILURE;
            }

            if (cudaFree(d_C) != cudaSuccess)
            {
                fprintf(stderr, "!!!! 1 memory free error (C)\n");
                //return EXIT_FAILURE;
            }
            if (detailedtime)
            {
            stoptime2 = omp_get_wtime();
//            printf("CUBLAS - Time for matrix multiplication: %12.3f s, for matrix size %d \n", (stoptime2-starttime2), N);
//            printf("Sec 1a completed\n");
            }

            status = cublasDestroy(handle);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "!!!! shutdown error (A)\n");
                //return EXIT_FAILURE;
            }


	    }

        #pragma omp section
	    {
            tid = omp_get_thread_num();
            //printf("Sec2 -- Hello World from thread = %d\n", tid);
            nthreads = omp_get_num_threads();
            //printf(" Sec 1b \n -- The number of threads for mkl %d \n", mkl_get_max_threads());

	        float *d_A = 0;
            float *d_B = 0;
            float *d_C = 0;
            cublasHandle_t handle;



            checkCudaErrors(cudaSetDevice(1));
            //cublasInit();
//            printf("\n 2 Dev Number : %d\n",1);

             status = cublasCreate(&handle);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "!!!! 2 CUBLAS initialization error\n");
                //return EXIT_FAILURE;
            }


                     /* Allocate device memory for the matrices */
            if (cudaMalloc((void **)&d_A, n2 * sizeof(d_A[0])) != cudaSuccess)
            {
                fprintf(stderr, "!!!! 2 device memory allocation error (allocate A)\n");
                //return EXIT_FAILURE;
            }

            if (cudaMalloc((void **)&d_B, n2 * sizeof(d_B[0])) != cudaSuccess)
            {
                fprintf(stderr, "!!!! 2 device memory allocation error (allocate B)\n");
                //return EXIT_FAILURE;
            }

            if (cudaMalloc((void **)&d_C, n2 * sizeof(d_C[0])) != cudaSuccess)
            {
                fprintf(stderr, "!!!! 2 device memory allocation error (allocate C)\n");
                //return EXIT_FAILURE;
            }

            if (detailedtime)
            {
            starttime = omp_get_wtime();
            }
            /* Initialize the device matrices with the host matrices */

            starttime_vect = omp_get_wtime();

            status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "!!!! 2 device access error (write B)\n");
                //return EXIT_FAILURE;
            }

           // status = cublasSetVector(n2, sizeof(cublas_C[0]), cublas_C, 1, d_C, 1);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "!!!! 2 device access error (write C)\n");
                //return EXIT_FAILURE;
            }
            stoptime_vect = omp_get_wtime();

            //for 2nd half of Cublas
            status = cublasSetVector(n2*(1-scale), sizeof(h_A[0]), h_A+(int)(m*(scale)*p), 1, d_A, 1);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "!!!! 2 device access error (2 write A)\n");
               // return EXIT_FAILURE;
            }
            //status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m*(1-scale), k, &alpha, d_B, ldb, d_A+(int)(m*(1-scale)*p), lda, &beta, d_C+(int)(m*(1-scale)*n), ldc);
            status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m*(1-scale), k, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc);

            // TODO read - http://stackoverflow.com/questions/14595750/transpose-matrix-multiplication-in-cublas-howto

            /* Read the result back */
            status = cublasGetVector(n2*(1-scale), sizeof(cublas_C[0]), d_C, 1, cublas_C+(int)(m*(scale)*n), 1);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "!!!! 2 device access error (2 read C)\n");
               // return EXIT_FAILURE;
            }
            if (detailedtime)
            {
            stoptime = omp_get_wtime();

//            printf("CUBLAS - Time for matrix multiplication: %12.3f s, for threads %d , for matrix size %d \n", stoptime-starttime, atoi(argv[1]), N);
//            printf("Sec 1b completed\n");
            }

            if (cudaFree(d_A) != cudaSuccess)
            {
                fprintf(stderr, "!!!! 2 memory free error (A)\n");
                //return EXIT_FAILURE;
            }

            if (cudaFree(d_B) != cudaSuccess)
            {
                fprintf(stderr, "!!!! 2 memory free error (B)\n");
                //return EXIT_FAILURE;
            }

            if (cudaFree(d_C) != cudaSuccess)
            {
                fprintf(stderr, "!!!! 2 memory free error (C)\n");
                //return EXIT_FAILURE;
            }

            status = cublasDestroy(handle);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "!!!! shutdown error (A)\n");
               // return EXIT_FAILURE;
            }
	    }
    }
    stage1stop = omp_get_wtime();
//    printf("CUBLAS on dev 0 (%2.3f) + CUBLAS on dev 1 (%2.3f) - Total Time for matrix multiplication:$ %12.3f $s, for threads %d , for matrix size %d x %d \n",scale, 1-scale, (stage1stop-stage1start)+(stoptime_vect-starttime_vect), mkl_get_max_threads(), N,N);
    printf("CUBLAS on dev 0 (%2.3f) + CUBLAS on dev 1 (%2.3f) - Total Time for matrix multiplication:$ %12.3f $s, for threads %d , for matrix size %d x %d \n",scale, 1-scale, MAX((stoptime-starttime),(stoptime2-starttime2)), mkl_get_max_threads(), N,N);
    printf("CUBLAS on dev 0 (%2.3f) + CUBLAS on dev 1 (%2.3f) - Total Time for matrix multiplication:$ %12.3f $s, for threads %d , for matrix size %d x %d \n",1-scale,scale, MAX((stoptime-starttime),(stoptime2-starttime2)), mkl_get_max_threads(), N,N);

    // Stage 2 - 1st half mkl, 2nd half cublas
    // This stage stores the result matrix in mkl_C
//    printf("Stage 2\n");
//    stage2start = omp_get_wtime();
//    #pragma omp parallel sections private(nthreads, tid) num_threads(2)
//    {
//    	 /* Obtain thread number */
//	    tid = omp_get_thread_num();
//	  //  printf("Hello World from thread = %d\n", tid);
//
//	    // /* Only master thread does this */
//	    if (tid == 0)
//	    {
//	      nthreads = omp_get_num_threads();
//	    }
//        #pragma omp section
//	    {
//            tid = omp_get_thread_num();
//            //printf("Sec 2a \n -- The number of threads for mkl %d \n", mkl_get_max_threads());
//            nthreads = omp_get_num_threads();
//
//            //Warming up
//            // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m*(scale), n, p, alpha, h_A, p, h_B, n, beta, mkl_C, n);
//            // Warming up done
//
//            if (detailedtime)
//            {
//            starttime = omp_get_wtime();
//            }
//
//            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m*(scale), n, p, alpha, h_A, p, h_B, n, beta, mkl_C, n);
//
//            if (detailedtime)
//            {
//            stoptime = omp_get_wtime();
//            printf("MKL - Time for matrix multiplication: %12.3f s, for threads %d , for matrix size %d \n", stoptime-starttime, atoi(argv[1]), N);
//            printf("Sec 2a completed\n");
//            }
//
//	    }
//
//        #pragma omp section
//	    {
//            tid = omp_get_thread_num();
//            //printf("Sec2 ");
//            nthreads = omp_get_num_threads();
//
//            //printf("Sec2 Number of threads = %d\n", nthreads);
//            if (detailedtime)
//            {
//            starttime2 = omp_get_wtime();
//            }
//            //for 2nd half of Cublas
//            status = cublasSetVector(n2*(1-scale), sizeof(h_A[0]), h_A+(int)(m*(scale)*p), 1, d_A, 1);
//
//            if (status != CUBLAS_STATUS_SUCCESS)
//            {
//                fprintf(stderr, "!!!! device access error (2 write A)\n");
//               // return EXIT_FAILURE;
//            }
//            //status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m*(1-scale), k, &alpha, d_B, ldb, d_A+(int)(m*(1-scale)*p), lda, &beta, d_C+(int)(m*(1-scale)*n), ldc);
//            status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m*(1-scale), k, &alpha, d_B, ldb, d_A, lda, &beta, d_C, ldc);
//
//            // TODO read - http://stackoverflow.com/questions/14595750/transpose-matrix-multiplication-in-cublas-howto
//
//            /* Read the result back */
//            status = cublasGetVector(n2*(1-scale), sizeof(mkl_C[0]), d_C, 1, mkl_C+(int)(m*(scale)*n), 1);
//
//            if (status != CUBLAS_STATUS_SUCCESS)
//            {
//                fprintf(stderr, "!!!! device access error (2 read C)\n");
//               // return EXIT_FAILURE;
//            }
//
//            if (detailedtime)
//            {
//            stoptime2 = omp_get_wtime();
//            printf("CUBLAS - Time for matrix multiplication: %12.3f s, for matrix size %d \n",(stoptime2-starttime2)+(stoptime_vect-starttime_vect), N);
//            printf("Sec2 completed\n");
//            }
//
//	    }
//    }
//    stage2stop = omp_get_wtime();
//    printf("MKL (%2.3f) + CUBLAS (%2.3f) - Total Time for matrix multiplication:$ %12.3f $s, for threads %d , for matrix size %d x %d \n",scale, 1-scale, (stage2stop-stage2start)+(stoptime_vect-starttime_vect), mkl_get_max_threads(), N, N);

    /* Performs operation using cublas */

   if(debug)
   {
    printf("\n Matrix A \n");
    for (i = 0; i < min(16,n2); i++)
    {
      printf(" %3.3f ", h_A[i]);
    }

    printf("\n Matrix B \n");
    for (i = 0; i < min(16,n2); i++)
    {
      printf(" %3.3f ", h_B[i]);
    }

    printf("\nCUBLAS -  Matrix C  \n");
    for (i = 0; i < min(16,n2); i++)
    {
      printf(" %3.3f ", cublas_C[i]);
    }
   }
    /* Adding MKL code from here */

        /* Performs operation using MKL */



    /* Check result against reference */
    error_norm = 0;
    ref_norm = 0;

    for (i = 0; i < n2; ++i)
    {
        diff = mkl_C[i] - cublas_C[i];
        error_norm += diff * diff;
        ref_norm += mkl_C[i] * mkl_C[i];
    }

    error_norm = (float)sqrt((double)error_norm);
    ref_norm = (float)sqrt((double)ref_norm);

    if (fabs(ref_norm) < 1e-7)
    {
        fprintf(stderr, "!!!! reference norm is 0\n");
      //  return EXIT_FAILURE;
    }

   if(debug)
   {
    printf("\nMKL -  Matrix C  \n");
    for (i = 0; i < min(16,n2); i++)
    {
      printf(" %3.3f ", mkl_C[i]);
    }
   }
    printf(" Error norm : %3.3f \n",error_norm);
    printf(" Ref norm : %3.3f \n",ref_norm);

    /* mkl code ends here */

    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(cublas_C);
    free(mkl_C);

//    if (cudaFree(d_A) != cudaSuccess)
//    {
//        fprintf(stderr, "!!!! memory free error (A)\n");
//        return EXIT_FAILURE;
//    }
//
//    if (cudaFree(d_B) != cudaSuccess)
//    {
//        fprintf(stderr, "!!!! memory free error (B)\n");
//        return EXIT_FAILURE;
//    }
//
//    if (cudaFree(d_C) != cudaSuccess)
//    {
//        fprintf(stderr, "!!!! memory free error (C)\n");
//        return EXIT_FAILURE;
//    }

    /* Shutdown */
//    status = cublasDestroy(handle);
//
//    if (status != CUBLAS_STATUS_SUCCESS)
//    {
//        fprintf(stderr, "!!!! shutdown error (A)\n");
//        return EXIT_FAILURE;
//    }
    exit(error_norm / ref_norm < 1e-6f ? EXIT_SUCCESS : EXIT_FAILURE);
}
