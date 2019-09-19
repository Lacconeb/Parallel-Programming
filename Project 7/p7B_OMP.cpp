#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <omp.h>



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

	fp = fopen("p7_OMP_Results_1.txt", "w");

	int NUMT = 1;

	omp_set_num_threads( NUMT );

	int numTrials = 10;
	double avg_perf = 0;
	double peak_perf = 0;

	for(int j = 0; j < numTrials; j++)
	{

		double time0 = omp_get_wtime();

		#pragma omp parallel for default(none) shared(size, Array, Sums)
		for( int shift = 0; shift < size; shift++ )
		{
			float sum = 0.;
			for( int i = 0; i < size; i++ )
			{
				sum += Array[i] * Array[i + shift];
			}
			Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
		}

		double time1 = omp_get_wtime();
		double gigaTrialsPerSecond = (double)size * size / ( time1 - time0 ) / 1000000000.;
		if( gigaTrialsPerSecond > peak_perf )
			peak_perf = gigaTrialsPerSecond;
		avg_perf += gigaTrialsPerSecond;

	}

	printf("Threads = %d\n", NUMT);
	printf("Peak Performance = %lf\n", peak_perf);
	printf("Average Performance = %lf\n", avg_perf/numTrials);

	fprintf(fp, "Threads, Peak Performance\n");
	fprintf(fp, "%d,%lf\n", NUMT, peak_perf);

	fprintf(fp, "Index,Sum\n");
	for(int i = 0; i<size; i++)
		fprintf(fp, "%d,%f\n", i, Sums[i] );

	fclose(fp);

	return 0;
}