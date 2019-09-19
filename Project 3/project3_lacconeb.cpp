/*
CS 475 - Project 3
Author: Brian Laccone
Email: lacconeb@oregonstate.edu
Date: 5/6/2019

*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <omp.h>

int	NowYear;		// 2019 - 2024
int EndYear;		// 2025
int	NowMonth;		// 0 - 11

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
int		NowNumDeer;		// number of deer in the current population
float	NowBFGrainEffect;
int		NowBFDeerEffect;

const float GRAIN_GROWS_PER_MONTH =		8.0;
const float ONE_DEER_EATS_PER_MONTH =	0.5;

const float AVG_PRECIP_PER_MONTH =		6.0;	// average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =				2.0;	// plus or minus noise

const float AVG_TEMP =					50.0;	// average
const float AMP_TEMP =					20.0;	// plus or minus
const float RANDOM_TEMP =				10.0;	// plus or minus noise

const float MIDTEMP =					40.0;
const float MIDPRECIP =					10.0;



void SetTemp();
void SetPrec();

float Ranf( unsigned int *seedp,  float low, float high );
int Rani( unsigned int *seedp, int ilow, int ihigh );
float SQR( float x );

void GrainDeer();
void Grain();
void Watcher();

void Bigfoot();



int main()
{

	// starting date and time:
	NowMonth =    0;
	NowYear  = 2019;
	EndYear  = 2025;

	// starting state:
	NowNumDeer = 1;
	NowHeight =  2.;


	float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

	float temp = AVG_TEMP - AMP_TEMP * cos( ang );
	unsigned int seed = 0;
	NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

	float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
	NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
	if( NowPrecip < 0. )
		NowPrecip = 0.;

	omp_set_num_threads( 4 );	// same as # of sections
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			GrainDeer( );
		}

		#pragma omp section
		{
			Grain( );
		}

		#pragma omp section
		{
			Watcher( );
		}

		#pragma omp section
		{
			Bigfoot( );	// your own
		}

	}       // implied barrier -- all functions must return in order
		// to allow any of them to get past here
}

void GrainDeer()
{
	int deerTemp;

	while(NowYear < EndYear)
	{
		deerTemp = NowNumDeer;
		if((float) deerTemp < NowHeight)
			deerTemp++;
		else if ((float) deerTemp > NowHeight)
			deerTemp--;

		deerTemp += NowBFDeerEffect;
		// DoneComputing barrier:
		#pragma omp barrier

		NowNumDeer = deerTemp;

		// DoneAssigning barrier:
		#pragma omp barrier

		// DonePrinting barrier:
		#pragma omp barrier
	}
}

void Grain()
{
	float heightTemp;
	float tempFactor;
	float precipFactor;



	while(NowYear < EndYear)
	{
		heightTemp = NowHeight;
		tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );
		precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );
		heightTemp += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
		heightTemp -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;

		heightTemp += NowBFGrainEffect;


		if(heightTemp < 0)
			heightTemp = 0;

		// DoneComputing barrier:
		#pragma omp barrier

		NowHeight = heightTemp;

		// DoneAssigning barrier:
		#pragma omp barrier

		// DonePrinting barrier:
		#pragma omp barrier
	}
}

void Watcher()
{

	FILE *res = fopen("p3_results.csv", "w+");
	fprintf(res, "Month,Temperature (c),Precipitation (cm),Height (cm),Deer,Bigfoot Grain Effect,Bigfoot Deer Effect\n");

	float tempC;
	float heightTempCM;
	float precipTempCM;
	int monthTemp;

	unsigned int seed = 0;

	while(NowYear < EndYear)
	{
		// DoneComputing barrier:
		#pragma omp barrier

		// DoneAssigning barrier:
		#pragma omp barrier

		monthTemp = NowMonth + 12 * (NowYear - 2019);
		tempC = (5. / 9.) * (NowTemp - 32);
		heightTempCM = 2.54 * NowHeight;
		precipTempCM = 2.54 * NowPrecip;

		fprintf(res,"%d,%.2f,%.2f,%.2f,%d,%.2f,%d\n", monthTemp, tempC, precipTempCM, heightTempCM, NowNumDeer, NowBFGrainEffect, NowBFDeerEffect);

		if(NowMonth == 11)
		{
			NowMonth = 0;
			NowYear++;
		}
		else
		{
			NowMonth++;
		}


		float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

		float temp = AVG_TEMP - AMP_TEMP * cos( ang );
		NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

		float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
		NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
		if( NowPrecip < 0. )
			NowPrecip = 0.;

		// DonePrinting barrier:
		#pragma omp barrier


	}

	fclose(res);
}

void Bigfoot()
{
	int deerEffect;
	float grainEffect;

	while(NowYear < EndYear)
	{
		if(NowHeight > NowNumDeer)
		{
			grainEffect = -3.;
			deerEffect = 1;
		}
		else if(NowHeight < NowNumDeer)
		{
			grainEffect = 3.;
			deerEffect = -1;
		}
		else
		{
			grainEffect = 3.;
			deerEffect = 1;
		}

		// DoneComputing barrier:
		#pragma omp barrier

		NowBFGrainEffect = grainEffect;
		NowBFDeerEffect = deerEffect;

		//printf("NowBFGrainEffect = %.2f // NowBFDeerEffect = %d\n", NowBFGrainEffect, NowBFDeerEffect);

		// DoneAssigning barrier:
		#pragma omp barrier

		// DonePrinting barrier:
		#pragma omp barrier
	}
}


float
Ranf( unsigned int *seedp,  float low, float high )
{
        float r = (float) rand_r( seedp );              // 0 - RAND_MAX

        return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}


int
Rani( unsigned int *seedp, int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = (float)ihigh + 0.9999f;

        return (int)(  Ranf(seedp, low,high) );
}

float
SQR( float x )
{
        return x*x;
}