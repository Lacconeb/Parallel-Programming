#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <omp.h>
#include "simd.p4.h"

int main()
{

#ifndef _OPENMP
	fprintf(stderr, "OpenMP is not working\n");
	return 1;
#endif

	FILE *fp = fopen("signal.txt", "r");

	if( fp == NULL )
	{
		fprintf( stderr, "Cannot open file 'signal.txt'\n" );
		exit( 1 );
	}

	int size;
	fscanf( fp, "%d", &size );

	float *Array = new float[ 2*size ];
	float *Sums  = new float[ 1*size ];

	for( int i = 0; i < size; i++ )
	{
		fscanf( fp, "%f", &Array[i] );
		Array[i+size] = Array[i];
	}

	fclose( fp );

	int numTrials = 10;
	double avg_perf = 0;
	double peak_perf = 0;

	for(int j = 0; j < numTrials; j++)
	{

		double time0 = omp_get_wtime();

		for( int shift = 0; shift < size; shift++ )
		{
			Sums[shift] = SimdMulSum(Array, &Array[shift], size);
		}

		double time1 = omp_get_wtime();
		double gigaTrialsPerSecond = (double)size * size / ( time1 - time0 ) / 1000000000.;
		if( gigaTrialsPerSecond > peak_perf )
			peak_perf = gigaTrialsPerSecond;
		avg_perf += gigaTrialsPerSecond;

	}

	fp = fopen("p7_SIMD_Results.txt", "w");

	printf("Peak Performance = %lf\n", peak_perf);
	printf("Average Performance = %lf\n", avg_perf/numTrials);

	fprintf(fp, "Peak Performance\n");
	fprintf(fp, "%lf\n", peak_perf);
	fprintf(fp, "Index,Sum\n");
	for(int i = 0; i<size; i++)
		fprintf(fp, "%d,%f\n", i, Sums[i] );


	fclose(fp);


	return 0;
}