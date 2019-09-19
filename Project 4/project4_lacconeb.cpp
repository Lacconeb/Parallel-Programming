#include <omp.h>
#include "simd.p4.h"


#ifndef ARRAYSIZE
#define ARRAYSIZE	36000000
#endif


#ifndef STARTSIZE
#define STARTSIZE	1000
#endif


#ifndef NUMTRIES
#define NUMTRIES	10
#endif



float Randf(float low, float high)
{
	float ranNum = (float)rand();

	return (low + ranNum * (high - low) / (float)RAND_MAX);
}

void Mul(float *A, float *B, float *C, int len)
{
	for(int i = 0; i < len; i++)
	{
		C[i] = A[i] * B[i];
	}
}

float MulSum(float *A, float *B, int len)
{

	float sum = 0.;

	for(int i = 0; i < len; i++)
	{
		sum += A[i] * B[i];
	}

	return sum;
}

int main()
{
	
#ifndef _OPENMP
	fprintf(stderr, "OpenMP is not working\n");
	return 1;
#endif


	float *A = new float[ARRAYSIZE];
	float *B = new float[ARRAYSIZE];
	float *C = new float[ARRAYSIZE];

	for(int i = 0; i < ARRAYSIZE; i++)
	{
		A[i] = 0.;
		B[i] = 0.;
		C[i] = 0.;
	}

	FILE *res = fopen("p4_results.csv", "w+");
	fprintf(res, "Size,SimdMul,Mul,SimdMulSum,MulSum,Simd Multiply Speedup,SIMD Multiply + Reduction\n");


	for(int size = STARTSIZE; size <= ARRAYSIZE; size *= 2)
	{
		
		double perfSIMDMul = 0.;
		double sumSIMDMul = 0.;
		double perfSIMDMulSum = 0.;
		double sumSIMDMulSum = 0.;
		double perfMul = 0.;
		double sumMul = 0.;
		double perfMulSum = 0.;
		double sumMulSum = 0.;

		for(int i = 0; i < NUMTRIES; i++){
			
			double t1;
			double t2;
			double currPerf;
			//SIMD Mul
			t1 = omp_get_wtime();
			SimdMul(A,B,C,ARRAYSIZE);
			t2 = omp_get_wtime();
			currPerf = (double)ARRAYSIZE/(t2 - t1)/1000000;
			if(perfSIMDMul < currPerf)
				perfSIMDMul = currPerf;
			sumSIMDMul += currPerf;

			//SIMD Reduce
			t1 = omp_get_wtime();
			SimdMulSum(A,B,ARRAYSIZE);
			t2 = omp_get_wtime();
			currPerf = (double)ARRAYSIZE/(t2 - t1)/1000000;
			if(perfSIMDMulSum < currPerf)
				perfSIMDMulSum = currPerf;
			sumSIMDMulSum += currPerf;

			//Mul
			t1 = omp_get_wtime();
			Mul(A,B,C, ARRAYSIZE);
			t2 = omp_get_wtime();
			currPerf = (double)ARRAYSIZE/(t2 - t1)/1000000;
			if(perfMul < currPerf)
				perfMul = currPerf;
			sumMul += currPerf;

			//Reduce
			t1 = omp_get_wtime();
			MulSum(A,B,ARRAYSIZE);
			t2 = omp_get_wtime();
			currPerf = (double)ARRAYSIZE/(t2 - t1)/1000000;
			if(perfMulSum < currPerf)
				perfMulSum = currPerf;
			sumMulSum += currPerf;

		}

		float mulSU = perfSIMDMul / perfMul;
		float mulSumSU = perfSIMDMulSum / perfMulSum;

		printf( "%d,%8.2lf,%8.2lf,%8.2lf,%8.2lf,%8.2lf,%8.2lf\n", size, perfSIMDMul, perfMul, perfSIMDMulSum, perfMulSum, mulSU, mulSumSU);
		fprintf(res, "%d,%8.2lf,%8.2lf,%8.2lf,%8.2lf,%8.2lf,%8.2lf\n", size, perfSIMDMul, perfMul, perfSIMDMulSum, perfMulSum, mulSU, mulSumSU);

	}

	fclose(res);

	return 0;
}