// Array multiplication: C = A * B:

// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"


#ifndef BLOCKSIZE
#define BLOCKSIZE		64		// number of threads per block
#endif

#ifndef SIZE
#define SIZE			512000	// array size
#endif

#ifndef NUMTRIALS
#define NUMTRIALS		100		// to make the timing more accurate
#endif

#ifndef TOLERANCE
#define TOLERANCE		0.00001f	// tolerance to relative error
#endif

// ranges for the random numbers:
const float XCMIN = 0.0;
const float XCMAX = 2.0;
const float YCMIN = 0.0;
const float YCMAX = 2.0;
const float RMIN = 0.5;
const float RMAX = 2.0;

// function prototypes:
float		Ranf(float, float);
int			Ranf(int, int);
void		TimeOfDaySeed();


// array multiplication (CUDA Kernel) on the device: C = A * B

__global__  void ArrayMul( float *A, float *B, float *C, float *D )
{
	
	
	unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;


	// randomize the location and radius of the circle:
	float xc = A[gid];
	float yc = B[gid];
	float  r = C[gid];

	// solve for the intersection using the quadratic formula:
	float a = 2.;
	float b = -2. * (xc + yc);
	float c = xc * xc + yc * yc - r * r;
	float d = b * b - 4. * a * c;

	if (d >= 0.) {
		// hits the circle:
		// get the first intersection:
		d = sqrt(d);
		float t1 = (-b + d) / (2. * a);	// time to intersect the circle
		float t2 = (-b - d) / (2. * a);	// time to intersect the circle
		float tmin = t1 < t2 ? t1 : t2;		// only care about the first intersection

		if (tmin >= 0.) {
			// where does it intersect the circle?
			float xcir = tmin;
			float ycir = tmin;

			// get the unitized normal vector at the point of intersection:
			float nx = xcir - xc;
			float ny = ycir - yc;
			float n = sqrt(nx * nx + ny * ny);
			nx /= n;	// unit vector
			ny /= n;	// unit vector

			// get the unitized incoming vector:
			float inx = xcir - 0.;
			float iny = ycir - 0.;
			float in = sqrt(inx * inx + iny * iny);
			inx /= in;	// unit vector
			iny /= in;	// unit vector

			// get the outgoing (bounced) vector:
			float dot = inx * nx + iny * ny;
			float outx = inx - 2. * nx * dot;	// angle of reflection = angle of incidence`
			float outy = iny - 2. * ny * dot;	// angle of reflection = angle of incidence`

			// find out if it hits the infinite plate:
			float t = (0. - ycir) / outy;

			if (t < 0.) {
				
			}
			else {
				D[gid] = 1;
			}
		}

		
	}

		
	

	/*
	__shared__ float prods[BLOCKSIZE];

	unsigned int numItems = blockDim.x;
	unsigned int tnum = threadIdx.x;
	unsigned int wgNum = blockIdx.x;
	unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

	prods[tnum] = A[gid] * B[gid];

	for (int offset = 1; offset < numItems; offset *= 2)
	{
		int mask = 2 * offset - 1;
		__syncthreads();
		if ((tnum & mask) == 0)
		{
			prods[tnum] += prods[tnum + offset];
		}
	}

	__syncthreads();
	if (tnum == 0)
		C[wgNum] = prods[0];
	*/
}


// main program:

int
main( int argc, char* argv[ ] )
{
	int dev = findCudaDevice(argc, (const char **)argv);

	TimeOfDaySeed();

	// allocate host memory:

	float * xcs = new float [ SIZE ];
	float * ycs = new float [ SIZE ];
	float * rs = new float [ SIZE ];
	float * hD = new float [ SIZE ];

	for( int i = 0; i < SIZE; i++ )
	{
		xcs[i] = Ranf(XCMIN, XCMAX);
		ycs[i] = Ranf(YCMIN, YCMAX);
		rs[i] = Ranf(RMIN, RMAX);
		hD[i] = 0;
	}

	// allocate device memory:

	float *dA, *dB, *dC, *dD;

	dim3 dimsA( SIZE, 1, 1 );
	dim3 dimsB( SIZE, 1, 1 );
	dim3 dimsC( SIZE, 1, 1 );
	dim3 dimsD( SIZE, 1, 1 );

	//__shared__ float prods[SIZE/BLOCKSIZE];


	cudaError_t status;
	status = cudaMalloc( reinterpret_cast<void **>(&dA), SIZE*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dB), SIZE*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dC), SIZE*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc(reinterpret_cast<void**>(&dD), SIZE * sizeof(float));
		checkCudaErrors(status);


	// copy host memory to the device:

	status = cudaMemcpy( dA, xcs, SIZE*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );
	status = cudaMemcpy( dB, ycs, SIZE*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );
	status = cudaMemcpy(dC, rs, SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaErrors(status);
	status = cudaMemcpy(dD, hD, SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCudaErrors(status);

	// setup the execution parameters:

	dim3 threads(BLOCKSIZE, 1, 1 );
	dim3 grid( SIZE / threads.x, 1, 1 );

	// Create and start timer

	cudaDeviceSynchronize( );

	// allocate CUDA events that we'll use for timing:

	cudaEvent_t start, stop;
	status = cudaEventCreate( &start );
		checkCudaErrors( status );
	status = cudaEventCreate( &stop );
		checkCudaErrors( status );

	// record the start event:

	status = cudaEventRecord( start, NULL );
		checkCudaErrors( status );

	// execute the kernel:

	for( int t = 0; t < NUMTRIALS; t++)
	{
	        ArrayMul<<< grid, threads >>>( dA, dB, dC, dD );
	}

	// record the stop event:

	status = cudaEventRecord( stop, NULL );
		checkCudaErrors( status );

	// wait for the stop event to complete:

	status = cudaEventSynchronize( stop );
		checkCudaErrors( status );

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime( &msecTotal, start, stop );
		checkCudaErrors( status );

	// compute and print the performance

	double secondsTotal = 0.001 * (double)msecTotal;
	double multsPerSecond = (float)SIZE * (float)NUMTRIALS / secondsTotal;
	double megaMultsPerSecond = multsPerSecond / 1000000.;
	fprintf( stderr, "Array Size = %10d, MegaMultReductions/Second = %10.2lf\n", SIZE, megaMultsPerSecond );

	// copy result from the device to the host:

	status = cudaMemcpy( hD, dD, SIZE*sizeof(float), cudaMemcpyDeviceToHost );
		checkCudaErrors( status );

	// check the sum :
	
	double sum = 0;
	for(int i = 0; i < SIZE; i++ )
	{
		sum += (double)hD[i];
	}
	float prob = (float)sum / (float)SIZE;

	fprintf(stderr, "\nsum = %lf\n", sum);
	fprintf( stderr, "\nprob = %lf\n", prob );
	

	// clean up memory:
	delete [ ] xcs;
	delete [ ] ycs;
	delete [ ] rs;
	delete [ ] hD;

	status = cudaFree( dA );
		checkCudaErrors( status );
	status = cudaFree( dB );
		checkCudaErrors( status );
	status = cudaFree( dC );
		checkCudaErrors( status );
	status = cudaFree( dD );
		checkCudaErrors(status);


	return 0;
}

float
Ranf(float low, float high)
{
	float r = (float)rand();               // 0 - RAND_MAX
	float t = r / (float)RAND_MAX;       // 0. - 1.

	return   low + t * (high - low);
}

int
Ranf(int ilow, int ihigh)
{
	float low = (float)ilow;
	float high = ceil((float)ihigh);

	return (int)Ranf(low, high);
}

void
TimeOfDaySeed()
{
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

	time_t  timer;
	time(&timer);
	double seconds = difftime(timer, mktime(&y2k));
	unsigned int seed = (unsigned int)(1000. * seconds);    // milliseconds
	srand(seed);
}