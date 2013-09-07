#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "global.h"

extern void jacobi(int);

// Array indexing macro
#define IX(i,j) ((i)+(N+2)*(j))

/*
	Set zero value in velocity, density, pressure and divergence. 
	Typically done at the start of simulation. 
	Had to do it this way because there are no memset function that
	work with pointers allocated using acc_malloc. 
	Also had separate it to a function because OpenACC appears to not like
	pointers in structs (e.g. config->pointer).
	This is not very efficient and there is probably a better way to do it. 
	
*/
void setAllMem(int N, float *velx, float *velx0, float *vely, float *vely0, float *dens, float *dens0, float *pres, float *pres0, float *div) {
	for (int i=0; i<(N+2)*(N+2); i++) {
		velx[i] = velx0[i] = 0;
		vely[i] = vely0[i] = 0;
		dens[i] = dens0[i] = 0;
		pres[i] = pres0[i] = 0;
		div[i] = 0;
	}	
}

/*
	Same as setAllMem function but here we only want to set zero value in one array.
	This is used before jacobi to set initial guess to 0. 
*/
void setMem(int N, float *d) {
	for (int i=0; i<(N+2)*(N+2); i++)
		d[i] = 0;
}

/*
	Here we can set the density within a range to convert it into a color scale. 
	Can also set a custom color to the density. 
	At the moment this does nothing but move the values from the density array 
	allocated on the accelerator to a output array of float4 allocated on the host 
	and is used for rendering rendered. 
	The output array is only copied from the device and never onto. 
*/
void densityToColor(unsigned char *output, float *density, int N) {
	for (int j=1; j<=N; j++) {
		for (int i=1; i<=N; i++) {
			// Just copy data from density to output right now
			// Might do some color manipulation later
			float densityValue = density[IX(i,j)];
			output[(j-1)*N + (i-1)] = (unsigned char)((densityValue/3.0)*255.0);
		}
	}
}

/*
	Fluid emitter that adds density to the simulation.
	- This method increases the density in a certain location
	- The added density is scaled using a timestep
*/
void addDensity ( int N, float *dens, float dt, float emposx, float emposy, float radius, float amount ) {
{
	for (int j=1; j<=N; j++) {
		for (int i=1; i<=N; i++) {
			// Emitt density in a circle
			if (((i-emposx)*(i-emposx)+(j-emposy)*(j-emposy)) < (radius*radius)) {
				dens[IX(i,j)] += dt * amount;
			}
		}
	}
}
}

/*
	Add buoyancy to the density
	- Bouyancy adds a directional velocity to the current velocity of the cell
	- Buoyance is equal to timestep * (strength of the buoyance) * (direction of buoyance) * (density amount at cell)
	- Velocity gets higher the more density there is in a cell
*/
void addDensityBuoyancy (int N, float *velx, float *vely, float *dens, float bdirx, float bdiry, float bstrength, float dt) {
{	
	for (int j=1; j<=N; j++) {
		for (int i=1; i<=N; i++) {
			velx[IX(i,j)] += dt * bstrength * bdirx * dens[IX(i,j)];
			vely[IX(i,j)] += dt * bstrength * bdiry * dens[IX(i,j)];
		}
	}
}
}

/*
	Advection of velocity and density
	- Can be used for advection of density or self-advection of velocity
	- Traces the cell velocity back in time and uses the previous value to effect the current value
	- To find the previous cell we have to interpolate
	- If the trace goes outside the grid we clamp to the edge
	- Density and velocity in the surrounding two celles of the grid are set to zero. this represents the border. 
	- Also dampen the velocity to get a gradual slowdown of the fluid in the simulation
*/
void advect (int N, float *d, float *d0, float *velx, float *vely, float dt, float vdamp, int b) {

	float dt0 = dt*N;
{
	for (int j=1; j<=N; j++) {
		for (int i=1; i<=N; i++) {
			// Calculate the traceback coordiantes
			float x = i - dt0 * velx[IX(i,j)]; 
			float y = j - dt0 * vely[IX(i,j)];		

			// Clamp to edge in x
			if (x<0.5f) x=0.5f; 
			if (x>N+0.5f) x=N+0.5f; 

			int i0=(int)x; 
			int i1=i0+1;

			// Clamp to edge in y
			if (y<0.5f) y=0.5f; 
			if (y>N+0.5f) y=N+0.5f;

			int j0=(int)y; 
			int j1=j0+1;

			// interpolation variables
			float s1 = x-i0; 
			float s0 = 1-s1; 
			float t1 = y-j0; 
			float t0 = 1-t1;

			// use the interpolation variables to get the previous velocity
			d[IX(i,j)] = (1.0f-vdamp*dt) * (s0 * (t0 * d0[IX(i0,j0)] + t1 * d0[IX(i0,j1)]) +
						 					s1 * (t0 * d0[IX(i1,j0)] + t1 * d0[IX(i1,j1)]));		
		}
	}

	// Set values on the borders to zero
	for (int i=1 ; i<=N ; i++) {
		d[IX(0,i)] = d[IX(N+1,i)] = d[IX(i,0)] = d[IX(i,N+1)] = 0.0f;
	}

	// Set the value in the corners to zero
	d[IX(0,0)] = d[IX(0,N+1)] = d[IX(N+1,0)] = d[IX(N+1,N+1)] = 0.0f;
}
}

/*
	Divergence step for solving the poisson-pressure equation for p (pressure)
	- Calculate the divergence field
	- Sample the neighbouring velocities
	- Set neighbouring velocity to zero if it is a edge cell
	- Calculate the divergence of the current cell by using the neighbouring values
	
*/
void divergence (int N, float *velx, float *vely, float *div) {
{	
	for (int j=1; j<=N; j++) {
		for (int i=1; i<=N; i++) {

			// Get neighbour velocities
			float vLx = velx[IX(i-1,j)];
			float vRx = velx[IX(i+1,j)];
			float vTy = vely[IX(i,j+1)];
			float vBy = vely[IX(i,j-1)];

			// Set neighbour velocity to
			// zero it if is a edge cell
			if (i==1) vLx = 0.0f;
			if (i==N) vRx = 0.0f;
			if (j==1) vBy = 0.0f;
			if (j==N) vTy = 0.0f;

			// Calculate divergence using finite difference
			div[IX(i,j)] = 0.5f * (vRx - vLx + vTy - vBy)/N;
		}
	}
}
}

/*
	Projection step
	- Poisson-pressure equation for p has already been solved in the jacobi step
	- Subtract the gradient to get a divergence-free velocity
	- Also use a normal vector to make the density slide at the edges of the grid
*/
void projection (int N, float *velx, float *vely, float *pres) {
{		
	for (int j=1; j<=N; j++) {
		for (int i=1; i<=N; i++) {

			// set normal vector component to zero if we are at a edge
			// normal x component to zero if we are at the left or right edges
			// normal y component to zero if we are at the top of bottom edges
			float normx=1.0f, normy=1.0f;
			if (i==1 || i==N) normx=0.0f;
			if (j==1 || j==N) normy=0.0f;

			// subtract the pressure gradient from the velocity field to get a 
			// divergence-free velocity
			velx[IX(i,j)] -= normx * 0.5f * N * (pres[IX(i+1,j)] - pres[IX(i-1,j)]);
			vely[IX(i,j)] -= normy * 0.5f * N * (pres[IX(i,j+1)] - pres[IX(i,j-1)]);				
		}
	}
}
}

/*
	Function for setting up and running the fluid simulation kernels
*/
void solveFluid(struct Configuration* config) { 
	// Jacobi settings
	float alpha = -1.0f;
	// Should use alpha -1 here, but this gives nicer results
	//float alpha = -(1.0f/invhalfgridscale);
	float rbeta = 0.25;
	int iterations = 100;

	// grid scaling. this is currently not used
    if(rank == 0){
        float gridscale = 1.0f;
        float invgridscale = 1.0f/gridscale;
        float invhalfgridscale = 0.5f/gridscale; 

        // Timstep value
        float timestep = 0.05f;

        // Emitter settings
        float amount = 2.0f;
        float radius = 0.5*config->N/10.0f;
        float emitterposx = config->N/2;
        float emitterposy = config->N/3;

        // buoyancy settings
        float bdiry = 1.0f;
        float bdirx = 0.0f;
        float bstrength = 0.1f;

        // advection settings
        float veldamp = 0.01f;	


        // Velocity advection
        float *tmp = config->velx0; config->velx0 = config->velx; config->velx = tmp;
        float *tmp1 = config->vely0; config->vely0 = config->vely; config->vely = tmp1;
        advect(config->N, config->velx, config->velx0, config->velx0, config->vely0, timestep, veldamp, 1);
        advect(config->N, config->vely, config->vely0, config->velx0, config->vely0, timestep, veldamp, 2);

        // Density advection
        float *tmp2 = config->dens0; config->dens0 = config->dens; config->dens = tmp2;
        advect(config->N, config->dens, config->dens0, config->velx, config->vely, timestep, 0.0f, 0);

        // Add density and density buoyancy
        addDensity(config->N, config->dens, timestep, emitterposx, emitterposy, radius, amount);
        addDensityBuoyancy(config->N, config->velx, config->vely, config->dens, bdirx, bdiry, bstrength, timestep);

        // Divergence calculation
        divergence(config->N, config->velx, config->vely, config->div);	

        // Pressure jacobi calculation. First set pres array to zero as initial guess
        setMem(config->N, config->pres);
    }

	jacobi(iterations);
	
    if(rank == 0){
        // Calculate projection
        projection(config->N, config->velx, config->vely, config->pres);	
    }
}

/*
	Allocate and initialize all the memory needed for the fluid simulation
*/
void initFluid(struct Configuration* config, int dimX, int dimY) { 

	// Set fluid dimensions. we set the grid size to 
	// +2 because we want the borders at the edge to be obstacles. 
	config->N = dimX;
	config->size = (dimX+2)*(dimY+2);

	// Allocate memory on the device using OpenACC runtime function acc_malloc. 
	// points allocated with acc_malloc has to be passed to kernels by using the
	// deviceptr() clause. 
	config->velx 	= (float *) malloc(sizeof(float)*config->size);
	config->velx0 	= (float *) malloc(sizeof(float)*config->size);
	config->vely 	= (float *) malloc(sizeof(float)*config->size);
	config->vely0 	= (float *) malloc(sizeof(float)*config->size);
	config->dens 	= (float *) malloc(sizeof(float)*config->size);
	config->dens0 	= (float *) malloc(sizeof(float)*config->size);
	config->pres 	= (float *) malloc(sizeof(float)*config->size);
	config->pres0 	= (float *) malloc(sizeof(float)*config->size);
	config->div 	= (float *) malloc(sizeof(float)*config->size);

	// Set an initial value of zero in all the arrays. 
	setAllMem(config->N, config->velx, config->velx0, config->vely, config->vely0, config->dens, config->dens0, config->pres, config->pres0, config->div);
}

/*
	Free allocated memory
*/
void freeFluid(struct Configuration *config ) { 

	free(config->velx); free(config->velx0);
	free(config->vely); free(config->vely0);
	free(config->dens); free(config->dens0);
	free(config->pres); free(config->pres0);
	free(config->div);	
}

/*
	Render fluid by running the densityToColor kernel.
*/
//void renderFluid(struct Configuration *config, float4 *output) { 
	//densityToColor(output, config->dens, config->N);
//}









