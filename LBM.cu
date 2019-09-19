/* This code accompanies
 *   Two relaxation time lattice Boltzmann method coupled to fast Fourier transform Poisson solver: Application to electroconvective flow, Journal of Computational Physics
 *	 https://doi.org/10.1016/j.jcp.2019.07.029
 *	 Three-dimensional Electro-convective Vortices in Cross-flow
 *	 https://arxiv.org/abs/1908.03861
 *   Yifei Guan, James Riley, Igor Novosselov
 *	 University of Washington
 *
 * Author: Yifei Guan
 *
 */
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda.h>
#include "LBM.h"
#include <cuda_runtime.h>
#define MAX(a, b) (((a) > (b)) ? (a) : (b)) 


__device__ __forceinline__ size_t gpu_field0_index(unsigned int x, unsigned int y, unsigned int z)
{
    return NX*(NY*z + y)+x;
}

__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y, unsigned int z)
{
    return NX*(NY*z + y)+x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int z, unsigned int d)
{
    return (NX*(NY*(NZ*(d-1)+z)+y)+x);
}

#define checkCudaErrors(err)  __checkCudaErrors(err,#err,__FILE__,__LINE__)
#define getLastCudaError(msg)  __getLastCudaError(msg,__FILE__,__LINE__)

inline void __checkCudaErrors(cudaError_t err, const char *const func, const char *const file, const int line )
{
    if(err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s(%d)\"%s\": [%d] %s.\n",
                file, line, func, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __getLastCudaError(const char *const errorMessage, const char *const file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s(%d): [%d] %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

// forward declarations of kernels
__global__ void gpu_initialization(double*, double*, double*, double*, double*, double*, double*, double*, double*);
//__global__ void gpu_taylor_green(unsigned int,double*,double*,double*);
__global__ void gpu_init_equilibrium(double*,double*,double*,double*,double*, double*, double*, double*, double*, double*, double*, double*);
__global__ void gpu_collide_save(double*,double*,double*,double*,double*,double*, double*, 
	double*, double*, double*, double*, double*, double*, double*, double,double*);
__global__ void gpu_boundary(double*, double*, double*, double*, double*, double*,double*);
__global__ void gpu_stream(double*, double*, double*, double*, double*, double*);
__global__ void gpu_bc_charge(double*, double*, double*);

__host__ void initialization(double *r, double *c, double *fi, double *u, double *v, double *w, double *ex, double *ey, double *ez)
{
	// blocks in grid
	dim3 grid(NX / nThreads, NY, NZ);

	// threads in block
	dim3 threads(nThreads, 1, 1);

	gpu_initialization << <grid, threads >> > (r, c, fi, u, v, w, ex, ey, ez);
	getLastCudaError("gpu_taylor_green kernel error");
}

__global__ void gpu_initialization(double *r, double *c, double *fi, double *u, double *v, double *w, double *ex, double *ey, double *ez)
{
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t sidx = gpu_scalar_index(x, y, z);
	r[sidx]  = rho0;
	c[sidx]  = 0.0;
	fi[sidx] = voltage * (Lz - dz*z) / Lz;
	u[sidx]  = 0.0;
	v[sidx]  = 0.0;
	w[sidx]  = 0.0;
	ex[sidx] = 0.0;
	ey[sidx] = 0.0;
	ez[sidx] = 0.0;
}

__host__ void init_equilibrium(double *f0, double *f1, double *h0, double *h1, double *r, double *c, 
								double *u, double *v, double *w, double *ex, double *ey, double *ez)
{
    // blocks in grid
    dim3  grid(NX/nThreads, NY, NZ);
    // threads in block
    dim3  threads(nThreads, 1, 1);

    gpu_init_equilibrium<<< grid, threads >>>(f0,f1,h0,h1,r,c,u,v,w,ex,ey,ez);
    getLastCudaError("gpu_init_equilibrium kernel error");
}

__global__ void gpu_init_equilibrium(double *f0, double *f1, double *h0, double *h1, double *r, double *c,
										double *u, double *v, double *w, double *ex, double *ey, double *ez)
{
    unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
    unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
    
    double rho    = r[gpu_scalar_index(x,y,z)];
    double ux     = u[gpu_scalar_index(x,y,z)];
    double uy     = v[gpu_scalar_index(x,y,z)];
	double uz     = w[gpu_scalar_index(x,y,z)];
	double charge = c[gpu_scalar_index(x,y,z)];
	double Ex     = ex[gpu_scalar_index(x,y,z)];
	double Ey     = ey[gpu_scalar_index(x,y,z)];
	double Ez     = ez[gpu_scalar_index(x,y,z)];

    // load equilibrium
    // feq_i  = w_i rho [1 + 3(ci . u) + (9/2) (ci . u)^2 - (3/2) (u.u)]
    // feq_i  = w_i rho [1 - 3/2 (u.u) + (ci . 3u) + (1/2) (ci . 3u)^2]
    // feq_i  = w_i rho [1 - 3/2 (u.u) + (ci . 3u){ 1 + (1/2) (ci . 3u) }]
    
    // temporary variables
    double w0r = w0*rho;
    double wsr = ws*rho;
	double war = wa*rho;
    double wdr = wd*rho;

	double w0c = w0*charge;
	double wsc = ws*charge;
	double wac = wa*charge;
	double wdc = wd*charge;

    double omusq   = 1.0 - 0.5*(ux*ux+uy*uy+uz*uz)/cs_square;
	double omusq_c = 1.0 - 0.5*((ux + K*Ex)*(ux + K*Ex) + (uy + K*Ey)*(uy + K*Ey) + (uz + K*Ez)*(uz + K*Ez)) / cs_square;
    
    double tux   = ux / cs_square / CFL;
    double tuy   = uy / cs_square / CFL;
	double tuz   = uz / cs_square / CFL;
	double tux_c = (ux + K*Ex) / cs_square / CFL;
	double tuy_c = (uy + K*Ey) / cs_square / CFL;
	double tuz_c = (uz + K*Ez) / cs_square / CFL;
    
	// zero weight
    f0[gpu_field0_index(x,y,z)]    = w0r*(omusq);
	h0[gpu_field0_index(x,y,z)]    = w0c*(omusq_c);
    
	// adjacent weight
	// flow
    double cidot3u = tux;
    f1[gpu_fieldn_index(x,y,z,1)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = -tux;
	f1[gpu_fieldn_index(x,y,z,2)]  = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
    cidot3u = tuy;
    f1[gpu_fieldn_index(x,y,z,3)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tuy;
    f1[gpu_fieldn_index(x,y,z,4)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tuz;
	f1[gpu_fieldn_index(x,y,z,5)] = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuz;
	f1[gpu_fieldn_index(x,y,z,6)] = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));

	// charge
	cidot3u = tux_c;
	h1[gpu_fieldn_index(x,y,z,1)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux_c;
	h1[gpu_fieldn_index(x,y,z,2)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c;
	h1[gpu_fieldn_index(x,y,z,3)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy_c;
	h1[gpu_fieldn_index(x,y,z,4)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz_c;
	h1[gpu_fieldn_index(x,y,z,5)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuz_c;
	h1[gpu_fieldn_index(x,y,z,6)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));

	// diagonal weight
	// flow
    cidot3u = tux+tuy;
    f1[gpu_fieldn_index(x,y,z,7)]  = war*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tuy-tux;
    f1[gpu_fieldn_index(x,y,z,8)]  = war*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tux+tuz;
    f1[gpu_fieldn_index(x,y,z,9)]  = war*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tux-tuz;
    f1[gpu_fieldn_index(x,y,z,10)] = war*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tuz + tuy;
	f1[gpu_fieldn_index(x,y,z,11)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy - tuz;
	f1[gpu_fieldn_index(x,y,z,12)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux - tuy;
	f1[gpu_fieldn_index(x,y,z,13)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy - tux;
	f1[gpu_fieldn_index(x,y,z,14)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux - tuz;
	f1[gpu_fieldn_index(x,y,z,15)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz - tux;
	f1[gpu_fieldn_index(x,y,z,16)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy - tuz;
	f1[gpu_fieldn_index(x,y,z,17)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz - tuy;
	f1[gpu_fieldn_index(x,y,z,18)] = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));

	// charge
	cidot3u = tux_c + tuy_c;
	h1[gpu_fieldn_index(x, y, z, 7)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy_c - tux_c;
	h1[gpu_fieldn_index(x, y, z, 8)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c + tuz_c;
	h1[gpu_fieldn_index(x, y, z, 9)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux_c - tuz_c;
	h1[gpu_fieldn_index(x, y, z, 10)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c + tuz_c;
	h1[gpu_fieldn_index(x, y, z, 11)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy_c - tuz_c;
	h1[gpu_fieldn_index(x, y, z, 12)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c - tuy_c;
	h1[gpu_fieldn_index(x, y, z, 13)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c - tux_c;
	h1[gpu_fieldn_index(x, y, z, 14)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c - tuz_c;
	h1[gpu_fieldn_index(x, y, z, 15)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz_c - tux_c;
	h1[gpu_fieldn_index(x, y, z, 16)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c - tuz_c;
	h1[gpu_fieldn_index(x, y, z, 17)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz_c - tuy_c;
	h1[gpu_fieldn_index(x, y, z, 18)] = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));


	// 3d diagonal
	//flow
	cidot3u = tux + tuy + tuz;
	f1[gpu_fieldn_index(x, y, z, 19)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy - tux - tuz;
	f1[gpu_fieldn_index(x, y, z, 20)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux + tuy - tuz;
	f1[gpu_fieldn_index(x, y, z, 21)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz - tux - tuy;
	f1[gpu_fieldn_index(x, y, z, 22)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux + tuz - tuy;
	f1[gpu_fieldn_index(x, y, z, 23)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy - tux - tuz;
	f1[gpu_fieldn_index(x, y, z, 24)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy + tuz - tux;
	f1[gpu_fieldn_index(x, y, z, 25)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux - tuy - tuz;
	f1[gpu_fieldn_index(x, y, z, 26)] = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));

	//charge
	cidot3u = tux_c + tuy_c + tuz_c;
	h1[gpu_fieldn_index(x, y, z, 19)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux_c -tuy_c - tuz_c;
	h1[gpu_fieldn_index(x, y, z, 20)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c + tuy_c - tuz_c;
	h1[gpu_fieldn_index(x, y, z, 21)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz_c - tux_c - tuy_c;
	h1[gpu_fieldn_index(x, y, z, 22)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c + tuz_c - tuy_c;
	h1[gpu_fieldn_index(x, y, z, 23)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c - tux_c - tuz_c;
	h1[gpu_fieldn_index(x, y, z, 24)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c + tuz_c - tux_c;
	h1[gpu_fieldn_index(x, y, z, 25)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c - tuy_c - tuz_c;
	h1[gpu_fieldn_index(x, y, z, 26)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));

}

__host__ void stream_collide_save(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2, double *r, double *c, 
	double *u, double *v, double *w, double *ex, double *ey, double *ez, double t,double *f0bc)
{
    // blocks in grid
    dim3  grid(NX/nThreads, NY, NZ);
    // threads in block
    dim3  threads(nThreads, 1, 1);

    gpu_collide_save<<< grid, threads >>>(f0,f1,f2, h0, h1, h2, r, c, u,v,w, ex, ey,ez,t,f0bc);
	gpu_boundary << < grid, threads >> >(f0, f1, f2, h0, h1, h2, f0bc);
	gpu_stream << < grid, threads >> >(f0, f1, f2, h0, h1, h2);
	gpu_bc_charge << < grid, threads >> >(h0, h1, h2);


    getLastCudaError("gpu_stream_collide_save kernel error");
}

__global__ void gpu_collide_save(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2, double *r, double *c,
	double *u, double *v, double *w, double *ex, double *ey, double *ez, double t,double *f0bc)
{
	// useful constants
	double omega_plus = 1.0 / (nu / cs_square / dt + 1.0 / 2.0) / dt;
	double omega_minus = 1.0 / (V / (nu / cs_square / dt) + 1.0 / 2.0) / dt;
	double omega_c_minus = 1.0 / (diffu / cs_square / dt + 1.0 / 2.0) / dt;
	double omega_c_minus = 1.0 / (VC / (diffu / cs_square / dt) + 1.0 / 2.0) / dt;

	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	// storage of f0 at upper and lower plate
	if (z == 0)  f0bc[gpu_field0_index(x, y, 0)] = f0[gpu_field0_index(x, y, z)];    // lower plate
	
	if (z==NZ-1) f0bc[gpu_field0_index(x, y, 1)] = f0[gpu_field0_index(x, y, z)]; // upper plate

	// load populations from nodes (ft is the same as f1)
	double ft0  = f0[gpu_field0_index(x, y, z)];
	double ht0  = h0[gpu_field0_index(x, y, z)];
	double ft1  = f1[gpu_fieldn_index(x, y, z, 1)];
	double ft2  = f1[gpu_fieldn_index(x, y, z, 2)];
	double ft3  = f1[gpu_fieldn_index(x, y, z, 3)];
	double ft4  = f1[gpu_fieldn_index(x, y, z, 4)];
	double ft5  = f1[gpu_fieldn_index(x, y, z, 5)];
	double ft6  = f1[gpu_fieldn_index(x, y, z, 6)];
	double ft7  = f1[gpu_fieldn_index(x, y, z, 7)];
	double ft8  = f1[gpu_fieldn_index(x, y, z, 8)];
	double ft9  = f1[gpu_fieldn_index(x, y, z, 9)];
	double ft10 = f1[gpu_fieldn_index(x, y, z, 10)];
	double ft11 = f1[gpu_fieldn_index(x, y, z, 11)];
	double ft12 = f1[gpu_fieldn_index(x, y, z, 12)];
	double ft13 = f1[gpu_fieldn_index(x, y, z, 13)];
	double ft14 = f1[gpu_fieldn_index(x, y, z, 14)];
	double ft15 = f1[gpu_fieldn_index(x, y, z, 15)];
	double ft16 = f1[gpu_fieldn_index(x, y, z, 16)];
	double ft17 = f1[gpu_fieldn_index(x, y, z, 17)];
	double ft18 = f1[gpu_fieldn_index(x, y, z, 18)];
	double ft19 = f1[gpu_fieldn_index(x, y, z, 19)];
	double ft20 = f1[gpu_fieldn_index(x, y, z, 20)];
	double ft21 = f1[gpu_fieldn_index(x, y, z, 21)];
	double ft22 = f1[gpu_fieldn_index(x, y, z, 22)];
	double ft23 = f1[gpu_fieldn_index(x, y, z, 23)];
	double ft24 = f1[gpu_fieldn_index(x, y, z, 24)];
	double ft25 = f1[gpu_fieldn_index(x, y, z, 25)];
	double ft26 = f1[gpu_fieldn_index(x, y, z, 26)];
	
	double ht1  = h1[gpu_fieldn_index(x, y, z, 1)];
	double ht2  = h1[gpu_fieldn_index(x, y, z, 2)];
	double ht3  = h1[gpu_fieldn_index(x, y, z, 3)];
	double ht4  = h1[gpu_fieldn_index(x, y, z, 4)];
	double ht5  = h1[gpu_fieldn_index(x, y, z, 5)];
	double ht6  = h1[gpu_fieldn_index(x, y, z, 6)];
	double ht7  = h1[gpu_fieldn_index(x, y, z, 7)];
	double ht8  = h1[gpu_fieldn_index(x, y, z, 8)];
	double ht9  = h1[gpu_fieldn_index(x, y, z, 9)];
	double ht10 = h1[gpu_fieldn_index(x, y, z, 10)];
	double ht11 = h1[gpu_fieldn_index(x, y, z, 11)];
	double ht12 = h1[gpu_fieldn_index(x, y, z, 12)];
	double ht13 = h1[gpu_fieldn_index(x, y, z, 13)];
	double ht14 = h1[gpu_fieldn_index(x, y, z, 14)];
	double ht15 = h1[gpu_fieldn_index(x, y, z, 15)];
	double ht16 = h1[gpu_fieldn_index(x, y, z, 16)];
	double ht17 = h1[gpu_fieldn_index(x, y, z, 17)];
	double ht18 = h1[gpu_fieldn_index(x, y, z, 18)];
	double ht19 = h1[gpu_fieldn_index(x, y, z, 19)];
	double ht20 = h1[gpu_fieldn_index(x, y, z, 20)];
	double ht21 = h1[gpu_fieldn_index(x, y, z, 21)];
	double ht22 = h1[gpu_fieldn_index(x, y, z, 22)];
	double ht23 = h1[gpu_fieldn_index(x, y, z, 23)];
	double ht24 = h1[gpu_fieldn_index(x, y, z, 24)];
	double ht25 = h1[gpu_fieldn_index(x, y, z, 25)];
	double ht26 = h1[gpu_fieldn_index(x, y, z, 26)];

	// compute macroscopic variables from microscopic variables
	double rho = ft0 + ft1 + ft2 + ft3 + ft4 + ft5 + ft6 + ft7 + ft8 
		+ ft9 + ft10 + ft11 + ft12 + ft13 + ft14 + ft15 + ft16 + ft17 + ft18 + ft19 + ft20 + ft21 + ft22 + ft23 + ft24 + ft25 + ft26;
	double rhoinv = 1.0 / rho;
	double charge = ht0 + ht1 + ht2 + ht3 + ht4 + ht5 + ht6 + ht7 + ht8 + ht9
		+ ht10 + ht11 + ht12 + ht13 + ht14 + ht15 + ht16 + ht17 + ht18 + ht19 + ht20 + ht21 + ht22 + ht23 + ht24 + ht25 + ht26;
	double Ex = ex[gpu_scalar_index(x, y, z)];
	double Ey = ey[gpu_scalar_index(x, y, z)];
	double Ez = ez[gpu_scalar_index(x, y, z)];
	double forcex = charge * Ex + exf;
	double forcey = charge * Ey;
	double forcez = charge * Ez;
	double ux = rhoinv*((ft1 + ft7 + ft9  + ft13 + ft15 + ft19 + ft21 + ft23 + ft26 
						- (ft2 + ft8 + ft10 + ft14 + ft16 + ft20 + ft22 + ft24 + ft25)) / CFL + forcex*dt*0.5);
	double uy = rhoinv*((ft3 + ft7 + ft11 + ft14 + ft17 + ft19 + ft21 + ft24 + ft25
						- (ft4 + ft8 + ft12 + ft13 + ft18 + ft20 + ft22 + ft23 + ft26)) / CFL + forcey*dt*0.5);
	double uz = rhoinv*((ft5 + ft9 + ft11 + ft16 + ft18 + ft19 + ft22 + ft23 + ft25
						- (ft6 + ft10 + ft12 + ft15 + ft17 + ft20 + ft21 + ft24 + ft26)) / CFL + forcez*dt*0.5);
	
	if (perturb==1){
		double xx = x*dx;
		double yy = y*dy;
		double zz = z*dz;
		// Rolling patterns
		//uz = (cos(2.0*M_PI*zz)-1)*cos(2*M_PI/LL*xx)*0.001;
		//ux = LL*sin(2.0*M_PI*zz)*sin(2*M_PI/LL*xx)*0.001;
		// Square patterns
		//uz = (cos(2 * M_PI*zz) - 1)*cos(2 * M_PI / LL*xx)*cos(2 * M_PI / LL*yy)*0.001;
		//ux = 0.5 * LL*sin(2 * M_PI*zz)*sin(2 * M_PI / LL*xx)*cos(2 * M_PI / LL*yy)*0.001;
		//uy = 0.5 * LL*sin(2 * M_PI*zz)*sin(2 * M_PI / LL*yy)*cos(2 * M_PI / LL*xx)*0.001;
		uz = uz+(cos(2 * M_PI*zz) - 1)*(cos(2 * M_PI / LL*xx)+cos(2 * M_PI / LL*yy))*0.001;
		ux = ux+LL*sin(2 * M_PI*zz)*sin(2 * M_PI / LL*xx)*0.001;
		uy = uy+LL*sin(2 * M_PI*zz)*sin(2 * M_PI / LL*yy)*0.001;

		// Hexagon patterns
		//double L = 1; // ratio of wavelength to domain size
		//double a = 4 * M_PI / 3 / L;
		//uz = -(cos(2 * M_PI*zz) - 1) / 3 
		//	*(2 * cos(2 * M_PI / (sqrtf(3)*L)*xx)*cos(2 * M_PI / (3 * L)*yy) + cos(4 * M_PI / (3 * L)*yy));
		//ux = -2 * M_PI*sin(2.0 * M_PI*zz)
		//	*(4.0 * M_PI) / (3 * sqrtf(3)*L*a *a)*sin(2 * M_PI / (sqrtf(3)*L)*xx)*cos(2 * M_PI / (3 * L)*yy);
		//uy = -2 * M_PI*sin(2.0 * M_PI*zz)*(4.0 * M_PI) / (9 * L*a *a)
		//	*(cos(2 * M_PI / (sqrtf(3)*L)*xx) + 2 * cos(2 * M_PI / (3 * L)*yy))*sin(2 * M_PI / (3 * L)*yy);
		//charge = charge + uz;

	}
	else{
		if (z == 0) {
			double ftm0 = f0[gpu_field0_index(x, y, 1)];
			double htm0 = h0[gpu_field0_index(x, y, 1)];
			double ftm1 = f1[gpu_fieldn_index(x, y, 1, 1)];
			double ftm2 = f1[gpu_fieldn_index(x, y, 1, 2)];
			double ftm3 = f1[gpu_fieldn_index(x, y, 1, 3)];
			double ftm4 = f1[gpu_fieldn_index(x, y, 1, 4)];
			double ftm5 = f1[gpu_fieldn_index(x, y, 1, 5)];
			double ftm6 = f1[gpu_fieldn_index(x, y, 1, 6)];
			double ftm7 = f1[gpu_fieldn_index(x, y, 1, 7)];
			double ftm8 = f1[gpu_fieldn_index(x, y, 1, 8)];
			double ftm9 = f1[gpu_fieldn_index(x, y, 1, 9)];
			double ftm10 = f1[gpu_fieldn_index(x, y, 1, 10)];
			double ftm11 = f1[gpu_fieldn_index(x, y, 1, 11)];
			double ftm12 = f1[gpu_fieldn_index(x, y, 1, 12)];
			double ftm13 = f1[gpu_fieldn_index(x, y, 1, 13)];
			double ftm14 = f1[gpu_fieldn_index(x, y, 1, 14)];
			double ftm15 = f1[gpu_fieldn_index(x, y, 1, 15)];
			double ftm16 = f1[gpu_fieldn_index(x, y, 1, 16)];
			double ftm17 = f1[gpu_fieldn_index(x, y, 1, 17)];
			double ftm18 = f1[gpu_fieldn_index(x, y, 1, 18)];
			double ftm19 = f1[gpu_fieldn_index(x, y, 1, 19)];
			double ftm20 = f1[gpu_fieldn_index(x, y, 1, 20)];
			double ftm21 = f1[gpu_fieldn_index(x, y, 1, 21)];
			double ftm22 = f1[gpu_fieldn_index(x, y, 1, 22)];
			double ftm23 = f1[gpu_fieldn_index(x, y, 1, 23)];
			double ftm24 = f1[gpu_fieldn_index(x, y, 1, 24)];
			double ftm25 = f1[gpu_fieldn_index(x, y, 1, 25)];
			double ftm26 = f1[gpu_fieldn_index(x, y, 1, 26)];

			double htm1 = h1[gpu_fieldn_index(x, y, 1, 1)];
			double htm2 = h1[gpu_fieldn_index(x, y, 1, 2)];
			double htm3 = h1[gpu_fieldn_index(x, y, 1, 3)];
			double htm4 = h1[gpu_fieldn_index(x, y, 1, 4)];
			double htm5 = h1[gpu_fieldn_index(x, y, 1, 5)];
			double htm6 = h1[gpu_fieldn_index(x, y, 1, 6)];
			double htm7 = h1[gpu_fieldn_index(x, y, 1, 7)];
			double htm8 = h1[gpu_fieldn_index(x, y, 1, 8)];
			double htm9 = h1[gpu_fieldn_index(x, y, 1, 9)];
			double htm10 = h1[gpu_fieldn_index(x, y, 1, 10)];
			double htm11 = h1[gpu_fieldn_index(x, y, 1, 11)];
			double htm12 = h1[gpu_fieldn_index(x, y, 1, 12)];
			double htm13 = h1[gpu_fieldn_index(x, y, 1, 13)];
			double htm14 = h1[gpu_fieldn_index(x, y, 1, 14)];
			double htm15 = h1[gpu_fieldn_index(x, y, 1, 15)];
			double htm16 = h1[gpu_fieldn_index(x, y, 1, 16)];
			double htm17 = h1[gpu_fieldn_index(x, y, 1, 17)];
			double htm18 = h1[gpu_fieldn_index(x, y, 1, 18)];
			double htm19 = h1[gpu_fieldn_index(x, y, 1, 19)];
			double htm20 = h1[gpu_fieldn_index(x, y, 1, 20)];
			double htm21 = h1[gpu_fieldn_index(x, y, 1, 21)];
			double htm22 = h1[gpu_fieldn_index(x, y, 1, 22)];
			double htm23 = h1[gpu_fieldn_index(x, y, 1, 23)];
			double htm24 = h1[gpu_fieldn_index(x, y, 1, 24)];
			double htm25 = h1[gpu_fieldn_index(x, y, 1, 25)];
			double htm26 = h1[gpu_fieldn_index(x, y, 1, 26)];

			// compute macroscopic variables from microscopic variables
			double rhom = ftm0 + ftm1 + ftm2 + ftm3 + ftm4 + ftm5 + ftm6 + ftm7 + ftm8
				+ ftm9 + ftm10 + ftm11 + ftm12 + ftm13 + ftm14 + ftm15 + ftm16 + ftm17 + ftm18 + ftm19 + ftm20 + ftm21 + ftm22 + ftm23 + ftm24 + ftm25 + ftm26;
			double rhoinvm = 1.0 / rho;
			double chargem = htm0 + htm1 + htm2 + htm3 + htm4 + htm5 + htm6 + htm7 + htm8 + htm9
				+ htm10 + htm11 + htm12 + htm13 + htm14 + htm15 + htm16 + htm17 + htm18 + htm19 + htm20 + htm21 + htm22 + htm23 + htm24 + htm25 + htm26;
			double Exm = ex[gpu_scalar_index(x, y, 1)];
			double Eym = ey[gpu_scalar_index(x, y, 1)];
			double Ezm = ez[gpu_scalar_index(x, y, 1)];
			double forcexm = chargem * Exm + exf;
			double forceym = chargem * Eym;
			double forcezm = chargem * Ezm;
			ux = -rhoinvm*((ftm1 + ftm7 + ftm9 + ftm13 + ftm15 + ftm19 + ftm21 + ftm23 + ftm26
				- (ftm2 + ftm8 + ftm10 + ftm14 + ftm16 + ftm20 + ftm22 + ftm24 + ftm25)) / CFL + forcexm*dt*0.5);
			uy = -rhoinvm*((ftm3 + ftm7 + ftm11 + ftm14 + ftm17 + ftm19 + ftm21 + ftm24 + ftm25
				- (ftm4 + ftm8 + ftm12 + ftm13 + ftm18 + ftm20 + ftm22 + ftm23 + ftm26)) / CFL + forceym*dt*0.5);
			uz = -rhoinvm*((ftm5 + ftm9 + ftm11 + ftm16 + ftm18 + ftm19 + ftm22 + ftm23 + ftm25
				- (ftm6 + ftm10 + ftm12 + ftm15 + ftm17 + ftm20 + ftm21 + ftm24 + ftm26)) / CFL + forcezm*dt*0.5);
		}
	}
	

	// write to memory (only when visualizing the data)
	
	r[gpu_scalar_index(x, y, z)] = rho;
	u[gpu_scalar_index(x, y, z)] = ux;
	v[gpu_scalar_index(x, y, z)] = uy;
	w[gpu_scalar_index(x, y, z)] = uz;
	c[gpu_scalar_index(x, y, z)] = charge;

	// collision step
	// now compute and relax to equilibrium
	// note that
	// feq_i  = w_i rho [1 + (ci . u / cs_square) + (1/2) (ci . u / cs_square)^2 - (1/2) (u.u) / cs_square]
	// feq_i  = w_i rho [1 - 1/2 (u.u)/cs_square + (ci . u / cs_square) + (1/2) (ci . u / cs_square)^2]
	// feq_i  = w_i rho [1 - 1/2 (u.u)/cs_square + (ci . u/cs_square){ 1 + (1/2) (ci . u/cs_square) }]
	// for charge transport equation, just change u into u + KE
	// heq_i  = w_i charge [1 - 1/2 (u.u)/cs_square + (ci . u/cs_square){ 1 + (1/2) (ci . u/cs_square) }]

	// choices of c
	// cx = [0, 1, 0, -1, 0, 1, -1, -1, 1] / CFL
	// cy = [0, 0, 1, 0, -1, 1, 1, -1, -1] / CFL

	// calculate equilibrium
	// temporary variables
	double w0r = w0*rho;
	double wsr = ws*rho;
	double war = wa*rho;
	double wdr = wd*rho;

	double w0c = w0*charge;
	double wsc = ws*charge;
	double wac = wa*charge;
	double wdc = wd*charge;

	double omusq = 1.0 - 0.5*(ux*ux + uy*uy + uz*uz) / cs_square;
	double omusq_c = 1.0 - 0.5*((ux + K*Ex)*(ux + K*Ex) + (uy + K*Ey)*(uy + K*Ey) + (uz + K*Ez)*(uz + K*Ez)) / cs_square;

	double tux = ux / cs_square / CFL;
	double tuy = uy / cs_square / CFL;
	double tuz = uz / cs_square / CFL;
	double tux_c = (ux + K*Ex) / cs_square / CFL;
	double tuy_c = (uy + K*Ey) / cs_square / CFL;
	double tuz_c = (uz + K*Ez) / cs_square / CFL;

	// zero weight
	double fe0 = w0r*(omusq);
	double he0 = w0c*(omusq_c);

	// adjacent weight
	// flow
	double cidot3u = tux;
	double fe1 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux;
	double fe2 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy;
	double fe3 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy;
	double fe4 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz;
	double fe5 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuz;
	double fe6 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));

	// charge
	cidot3u = tux_c;
	double he1 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux_c;
	double he2 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c;
	double he3 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy_c;
	double he4 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz_c;
	double he5 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuz_c;
	double he6 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));

	// diagonal weight
	// flow
	cidot3u = tux + tuy;
	double fe7 = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy - tux;
	double fe8 = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux + tuz;
	double fe9 = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux - tuz;
	double fe10 = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz + tuy;
	double fe11 = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy - tuz;
	double fe12 = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux - tuy;
	double fe13 = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy - tux;
	double fe14 = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux - tuz;
	double fe15 = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz - tux;
	double fe16 = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy - tuz;
	double fe17 = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz - tuy;
	double fe18 = war*(omusq + cidot3u*(1.0 + 0.5*cidot3u));

	// charge
	cidot3u = tux_c + tuy_c;
	double he7 = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy_c - tux_c;
	double he8 = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c + tuz_c;
	double he9 = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux_c - tuz_c;
	double he10 = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c + tuz_c;
	double he11 = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy_c - tuz_c;
	double he12 = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c - tuy_c;
	double he13 = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c - tux_c;
	double he14 = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c - tuz_c;
	double he15 = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz_c - tux_c;
	double he16 = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c - tuz_c;
	double he17 = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz_c - tuy_c;
	double he18 = wac*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));

	// 3d diagonal
	//flow
	cidot3u = tux + tuy + tuz;
	double fe19 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy - tux - tuz;
	double fe20 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux + tuy - tuz;
	double fe21 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz - tux - tuy;
	double fe22 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux + tuz - tuy;
	double fe23 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy - tux - tuz;
	double fe24 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy + tuz - tux;
	double fe25 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux - tuy - tuz;
	double fe26 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));

	//charge
	cidot3u = tux_c + tuy_c + tuz_c;
	double he19 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux_c - tuy_c - tuz_c;
	double he20 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c + tuy_c - tuz_c;
	double he21 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuz_c - tux_c - tuy_c;
	double he22 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c + tuz_c - tuy_c;
	double he23 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c - tux_c - tuz_c;
	double he24 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c + tuz_c - tux_c;
	double he25 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c - tuy_c - tuz_c;
	double he26 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));

	// calculate force population
	// temperory variables
	double coe0 = w0 / cs_square;
	double coes = ws / cs_square;
	double coea = wa / cs_square;
	double coed = wd / cs_square;

	double cflinv = 1.0 / CFL;

	double fpop0 = -coe0*(ux*forcex + uy*forcey + uz*forcez);
	double cflinv2 = cflinv*cflinv / cs_square;

	double fpop1 = coes*(-uy*forcey - uz*forcez + ((cflinv - ux) + (cflinv2*ux))*forcex);
	double fpop2 = coes*(-uy*forcey - uz*forcez + ((-cflinv - ux) + (cflinv2*ux))*forcex);
	double fpop3 = coes*(-ux*forcex - uz*forcez + ((cflinv - uy) + (cflinv2*uy))*forcey);
	double fpop4 = coes*(-ux*forcex - uz*forcez + ((-cflinv - uy) + (cflinv2*uy))*forcey);
	double fpop5 = coes*(-ux*forcex - uy*forcey + ((cflinv - uz) + (cflinv2*uz))*forcez);
	double fpop6 = coes*(-ux*forcex - uy*forcey + ((-cflinv - uz) + (cflinv2*uz))*forcez);

	double fpop7  = coea*(((cflinv - ux)  + (ux + uy)*cflinv2)*forcex + ((cflinv - uy)  + (ux + uy)*cflinv2)*forcey - uz*forcez);
	double fpop8  = coea*(((-cflinv - ux) + (ux + uy)*cflinv2)*forcex + ((-cflinv - uy) + (ux + uy)*cflinv2)*forcey - uz*forcez);
	double fpop9  = coea*(((cflinv - ux)  + (ux + uz)*cflinv2)*forcex + ((cflinv - uz)  + (ux + uz)*cflinv2)*forcez - uy*forcey);
	double fpop10 = coea*(((-cflinv - ux) + (ux + uz)*cflinv2)*forcex + ((-cflinv - uz) + (ux + uz)*cflinv2)*forcez - uy*forcey);
	double fpop11 = coea*(((cflinv - uy)  + (uy + uz)*cflinv2)*forcey + ((cflinv - uz)  + (uy + uz)*cflinv2)*forcez - ux*forcex);
	double fpop12 = coea*(((-cflinv - uy) + (uy + uz)*cflinv2)*forcey + ((-cflinv - uz) + (uy + uz)*cflinv2)*forcez - ux*forcex);

	double fpop13 = coea*(((cflinv - ux) + (ux - uy)*cflinv2)*forcex + ((-cflinv - uy) + (-ux + uy)*cflinv2)*forcey - uz*forcez);
	double fpop14 = coea*(((-cflinv - ux) + (ux - uy)*cflinv2)*forcex + ((cflinv - uy) + (-ux + uy)*cflinv2)*forcey - uz*forcez);
	double fpop15 = coea*(((cflinv - ux) + (ux - uz)*cflinv2)*forcex + ((-cflinv - uz) + (-ux + uz)*cflinv2)*forcez - uy*forcey);
	double fpop16 = coea*(((-cflinv - ux) + (ux - uz)*cflinv2)*forcex + ((cflinv - uz) + (-ux + uz)*cflinv2)*forcez - uy*forcey);
	double fpop17 = coea*(((cflinv - uy) + (uy - uz)*cflinv2)*forcey + ((-cflinv - uz) + (-uy + uz)*cflinv2)*forcez - ux*forcex);
	double fpop18 = coea*(((-cflinv - uy) + (uy - uz)*cflinv2)*forcey + ((cflinv - uz) + (-uy + uz)*cflinv2)*forcez - ux*forcex);

	double fpop19 = coed*(((cflinv-ux)+(ux+uy+uz)*cflinv2)*forcex+((cflinv-uy)+(ux+uy+uz)*cflinv2)*forcey+((cflinv-uz)+(ux+uy+uz)*cflinv2)*forcez);
	double fpop20 = coed*(((-cflinv-ux)+(ux+uy+uz)*cflinv2)*forcex+((-cflinv-uy)+(ux+uy+uz)*cflinv2)*forcey+((-cflinv-uz)+(ux+uy+uz)*cflinv2)*forcez);
	double fpop21 = coed*(((cflinv-ux)+(ux+uy-uz)*cflinv2)*forcex+((cflinv-uy)+(ux+uy-uz)*cflinv2)*forcey+((-cflinv-uz)+(-ux-uy+uz)*cflinv2)*forcez);
	double fpop22 = coed*(((-cflinv-ux)+(ux+uy-uz)*cflinv2)*forcex+((-cflinv-uy)+(ux+uy-uz)*cflinv2)*forcey+((cflinv-uz)+(-ux-uy+uz)*cflinv2)*forcez);
	double fpop23 = coed*(((cflinv-ux)+(ux-uy+uz)*cflinv2)*forcex+((-cflinv-uy)+(-ux+uy-uz)*cflinv2)*forcey+((cflinv-uz)+(ux-uy+uz)*cflinv2)*forcez);
	double fpop24 = coed*(((-cflinv-ux)+(ux-uy+uz)*cflinv2)*forcex+((cflinv-uy)+(-ux+uy-uz)*cflinv2)*forcey+((-cflinv-uz)+(ux-uy+uz)*cflinv2)*forcez);
	double fpop25 = coed*(((-cflinv-ux)+(ux-uy-uz)*cflinv2)*forcex+((cflinv-uy)+(-ux+uy+uz)*cflinv2)*forcey+((cflinv-uz)+(-ux+uy+uz)*cflinv2)*forcez);
	double fpop26 = coed*(((cflinv-ux)+(ux-uy-uz)*cflinv2)*forcex+((-cflinv-uy)+(-ux+uy+uz)*cflinv2)*forcey+((-cflinv-uz)+(-ux+uy+uz)*cflinv2)*forcez);
	
	// calculate f1 plus and minus
	double fp0 = ft0;
	double fp1 = 0.5*(ft1 + ft2);
	double fp2 = fp1;
	double fp3 = 0.5*(ft3 + ft4);
	double fp4 = fp3;
	double fp5 = 0.5*(ft5 + ft6);
	double fp6 = fp5;
	double fp7 = 0.5*(ft7 + ft8);
	double fp8 = fp7;
	double fp9 = 0.5*(ft9 + ft10);
	double fp10 = fp9;
	double fp11 = 0.5*(ft11 + ft12);
	double fp12 = fp11;
	double fp13 = 0.5*(ft13 + ft14);
	double fp14 = fp13;
	double fp15 = 0.5*(ft15 + ft16);
	double fp16 = fp15;
	double fp17 = 0.5*(ft17 + ft18);
	double fp18 = fp17;
	double fp19 = 0.5*(ft19 + ft20);
	double fp20 = fp19;
	double fp21 = 0.5*(ft21 + ft22);
	double fp22 = fp21;
	double fp23 = 0.5*(ft23 + ft24);
	double fp24 = fp23;
	double fp25 = 0.5*(ft25 + ft26);
	double fp26 = fp25;

	double fm0 = 0.0;
	double fm1 = 0.5*(ft1 - ft2);
	double fm2 = -fm1;
	double fm3 = 0.5*(ft3 - ft4);
	double fm4 = -fm3;
	double fm5 = 0.5*(ft5 - ft6);
	double fm6 = -fm5;
	double fm7 = 0.5*(ft7 - ft8);
	double fm8 = -fm7;
	double fm9 = 0.5*(ft9 - ft10);
	double fm10 = -fm9;
	double fm11 = 0.5*(ft11 - ft12);
	double fm12 = -fm11;
	double fm13 = 0.5*(ft13 - ft14);
	double fm14 = -fm13;
	double fm15 = 0.5*(ft15 - ft16);
	double fm16 = -fm15;
	double fm17 = 0.5*(ft17 - ft18);
	double fm18 = -fm17;
	double fm19 = 0.5*(ft19 - ft20);
	double fm20 = -fm19;
	double fm21 = 0.5*(ft21 - ft22);
	double fm22 = -fm21;
	double fm23 = 0.5*(ft23 - ft24);
	double fm24 = -fm23;
	double fm25 = 0.5*(ft25 - ft26);
	double fm26 = -fm25;

	// calculate feq plus and minus
	double fep0 = fe0;
	double fep1 = 0.5*(fe1 + fe2);
	double fep2 = fep1;
	double fep3 = 0.5*(fe3 + fe4);
	double fep4 = fep3;
	double fep5 = 0.5*(fe5 + fe6);
	double fep6 = fep5;
	double fep7 = 0.5*(fe7 + fe8);
	double fep8 = fep7;
	double fep9 = 0.5*(fe9 + fe10);
	double fep10 = fep9;
	double fep11 = 0.5*(fe11 + fe12);
	double fep12 = fep11;
	double fep13 = 0.5*(fe13 + fe14);
	double fep14 = fep13;
	double fep15 = 0.5*(fe15 + fe16);
	double fep16 = fep15;
	double fep17 = 0.5*(fe17 + fe18);
	double fep18 = fep17;
	double fep19 = 0.5*(fe19 + fe20);
	double fep20 = fep19;
	double fep21 = 0.5*(fe21 + fe22);
	double fep22 = fep21;
	double fep23 = 0.5*(fe23 + fe24);
	double fep24 = fep23;
	double fep25 = 0.5*(fe25 + fe26);
	double fep26 = fep25;

	double fem0 = 0.0;
	double fem1 = 0.5*(fe1 - fe2);
	double fem2 = -fem1;
	double fem3 = 0.5*(fe3 - fe4);
	double fem4 = -fem3;
	double fem5 = 0.5*(fe5 - fe6);
	double fem6 = -fem5;
	double fem7 = 0.5*(fe7 - fe8);
	double fem8 = -fem7;
	double fem9 = 0.5*(fe9 - fe10);
	double fem10 = -fem9;
	double fem11 = 0.5*(fe11 - fe12);
	double fem12 = -fem11;
	double fem13 = 0.5*(fe13 - fe14);
	double fem14 = -fem13;
	double fem15 = 0.5*(fe15 - fe16);
	double fem16 = -fem15;
	double fem17 = 0.5*(fe17 - fe18);
	double fem18 = -fem17;
	double fem19 = 0.5*(fe19 - fe20);
	double fem20 = -fem19;
	double fem21 = 0.5*(fe21 - fe22);
	double fem22 = -fem21;
	double fem23 = 0.5*(fe23 - fe24);
	double fem24 = -fem23;
	double fem25 = 0.5*(fe25 - fe26);
	double fem26 = -fem25;

	// calculate h1 plus and minus
	double hp0 = ht0;
	double hp1 = 0.5*(ht1 + ht2);
	double hp2 = hp1;
	double hp3 = 0.5*(ht3 + ht4);
	double hp4 = hp3;
	double hp5 = 0.5*(ht5 + ht6);
	double hp6 = hp5;
	double hp7 = 0.5*(ht7 + ht8);
	double hp8 = hp7;
	double hp9 = 0.5*(ht9 + ht10);
	double hp10 = hp9;
	double hp11 = 0.5*(ht11 + ht12);
	double hp12 = hp11;
	double hp13 = 0.5*(ht13 + ht14);
	double hp14 = hp13;
	double hp15 = 0.5*(ht15 + ht16);
	double hp16 = hp15;
	double hp17 = 0.5*(ht17 + ht18);
	double hp18 = hp17;
	double hp19 = 0.5*(ht19 + ht20);
	double hp20 = hp19;
	double hp21 = 0.5*(ht21 + ht22);
	double hp22 = hp21;
	double hp23 = 0.5*(ht23 + ht24);
	double hp24 = hp23;
	double hp25 = 0.5*(ht25 + ht26);
	double hp26 = hp25;

	double hm0 = 0.0;
	double hm1 = 0.5*(ht1 - ht2);
	double hm2 = -hm1;
	double hm3 = 0.5*(ht3 - ht4);
	double hm4 = -hm3;
	double hm5 = 0.5*(ht5 - ht6);
	double hm6 = -hm5;
	double hm7 = 0.5*(ht7 - ht8);
	double hm8 = -hm7;
	double hm9 = 0.5*(ht9 - ht10);
	double hm10 = -hm9;
	double hm11 = 0.5*(ht11 - ht12);
	double hm12 = -hm11;
	double hm13 = 0.5*(ht13 - ht14);
	double hm14 = -hm13;
	double hm15 = 0.5*(ht15 - ht16);
	double hm16 = -hm15;
	double hm17 = 0.5*(ht17 - ht18);
	double hm18 = -hm17;
	double hm19 = 0.5*(ht19 - ht20);
	double hm20 = -hm19;
	double hm21 = 0.5*(ht21 - ht22);
	double hm22 = -hm21;
	double hm23 = 0.5*(ht23 - ht24);
	double hm24 = -hm23;
	double hm25 = 0.5*(ht25 - ht26);
	double hm26 = -hm25;

	// calculate heq plus and minus
	double hep0 = he0;
	double hep1 = 0.5*(he1 + he2);
	double hep2 = hep1;
	double hep3 = 0.5*(he3 + he4);
	double hep4 = hep3;
	double hep5 = 0.5*(he5 + he6);
	double hep6 = hep5;
	double hep7 = 0.5*(he7 + he8);
	double hep8 = hep7;
	double hep9 = 0.5*(he9 + he10);
	double hep10 = hep9;
	double hep11 = 0.5*(he11 + he12);
	double hep12 = hep11;
	double hep13 = 0.5*(he13 + he14);
	double hep14 = hep13;
	double hep15 = 0.5*(he15 + he16);
	double hep16 = hep15;
	double hep17 = 0.5*(he17 + he18);
	double hep18 = hep17;
	double hep19 = 0.5*(he19 + he20);
	double hep20 = hep19;
	double hep21 = 0.5*(he21 + he22);
	double hep22 = hep21;
	double hep23 = 0.5*(he23 + he24);
	double hep24 = hep23;
	double hep25 = 0.5*(he25 + he26);
	double hep26 = hep25;

	double hem0 = 0.0;
	double hem1 = 0.5*(he1 - he2);
	double hem2 = -hem1;
	double hem3 = 0.5*(he3 - he4);
	double hem4 = -hem3;
	double hem5 = 0.5*(he5 - he6);
	double hem6 = -hem5;
	double hem7 = 0.5*(he7 - he8);
	double hem8 = -hem7;
	double hem9 = 0.5*(he9 - he10);
	double hem10 = -hem9;
	double hem11 = 0.5*(he11 - he12);
	double hem12 = -hem11;
	double hem13 = 0.5*(he13 - he14);
	double hem14 = -hem13;
	double hem15 = 0.5*(he15 - he16);
	double hem16 = -hem15;
	double hem17 = 0.5*(he17 - he18);
	double hem18 = -hem17;
	double hem19 = 0.5*(he19 - he20);
	double hem20 = -hem19;
	double hem21 = 0.5*(he21 - he22);
	double hem22 = -hem21;
	double hem23 = 0.5*(he23 - he24);
	double hem24 = -hem23;
	double hem25 = 0.5*(he25 - he26);
	double hem26 = -hem25;
	
	// calculate force_plus and force_minus
	double forcep0 = fpop0;
	double forcep1 = 0.5*(fpop1 + fpop2);
	double forcep2 = forcep1;
	double forcep3 = 0.5*(fpop3 + fpop4);
	double forcep4 = forcep3;
	double forcep5 = 0.5*(fpop5 + fpop6);
	double forcep6 = forcep5;
	double forcep7 = 0.5*(fpop7 + fpop8);
	double forcep8 = forcep7;
	double forcep9 = 0.5*(fpop9 + fpop10);
	double forcep10 = forcep9;
	double forcep11 = 0.5*(fpop11 + fpop12);
	double forcep12 = forcep11;
	double forcep13 = 0.5*(fpop13 + fpop14);
	double forcep14 = forcep13;
	double forcep15 = 0.5*(fpop15 + fpop16);
	double forcep16 = forcep15;
	double forcep17 = 0.5*(fpop17 + fpop18);
	double forcep18 = forcep17;
	double forcep19 = 0.5*(fpop19 + fpop20);
	double forcep20 = forcep19;
	double forcep21 = 0.5*(fpop21 + fpop22);
	double forcep22 = forcep21;
	double forcep23 = 0.5*(fpop23 + fpop24);
	double forcep24 = forcep23;
	double forcep25 = 0.5*(fpop25 + fpop26);
	double forcep26 = forcep25;
	
	double forcem0 = 0.0;
	double forcem1 = 0.5*(fpop1 - fpop2);
	double forcem2 = -forcem1;
	double forcem3 = 0.5*(fpop3 - fpop4);
	double forcem4 = -forcem3;
	double forcem5 = 0.5*(fpop5 - fpop6);
	double forcem6 = -forcem5;
	double forcem7 = 0.5*(fpop7 - fpop8);
	double forcem8 = -forcem7;
	double forcem9 = 0.5*(fpop9 - fpop10);
	double forcem10 = -forcem9;
	double forcem11 = 0.5*(fpop11 - fpop12);
	double forcem12 = -forcem11;
	double forcem13 = 0.5*(fpop13 - fpop14);
	double forcem14 = -forcem13;
	double forcem15 = 0.5*(fpop15 - fpop16);
	double forcem16 = -forcem15;
	double forcem17 = 0.5*(fpop17 - fpop18);
	double forcem18 = -forcem17;
	double forcem19 = 0.5*(fpop19 - fpop20);
	double forcem20 = -forcem19;
	double forcem21 = 0.5*(fpop21 - fpop22);
	double forcem22 = -forcem21;
	double forcem23 = 0.5*(fpop23 - fpop24);
	double forcem24 = -forcem23;
	double forcem25 = 0.5*(fpop25 - fpop26);
	double forcem26 = -forcem25;

	double sp = 1.0 - 0.5*dt*omega_plus;
	double sm = 1.0 - 0.5*dt*omega_minus;

	double source0 = sp*fpop0;
	double source1 = sp*forcep1 + sm*forcem1;
	double source2 = sp*forcep2 + sm*forcem2;
	double source3 = sp*forcep3 + sm*forcem3;
	double source4 = sp*forcep4 + sm*forcem4;
	double source5 = sp*forcep5 + sm*forcem5;
	double source6 = sp*forcep6 + sm*forcem6;
	double source7 = sp*forcep7 + sm*forcem7;
	double source8 = sp*forcep8 + sm*forcem8;
	double source9 = sp*forcep9 + sm*forcem9;
	double source10 = sp*forcep10 + sm*forcem10;
	double source11 = sp*forcep11 + sm*forcem11;
	double source12 = sp*forcep12 + sm*forcem12;
	double source13 = sp*forcep13 + sm*forcem13;
	double source14 = sp*forcep14 + sm*forcem14;
	double source15 = sp*forcep15 + sm*forcem15;
	double source16 = sp*forcep16 + sm*forcem16;
	double source17 = sp*forcep17 + sm*forcem17;
	double source18 = sp*forcep18 + sm*forcem18;
	double source19 = sp*forcep19 + sm*forcem19;
	double source20 = sp*forcep20 + sm*forcem20;
	double source21 = sp*forcep21 + sm*forcem21;
	double source22 = sp*forcep22 + sm*forcem22;
	double source23 = sp*forcep23 + sm*forcem23;
	double source24 = sp*forcep24 + sm*forcem24;
	double source25 = sp*forcep25 + sm*forcem25;
	double source26 = sp*forcep26 + sm*forcem26;
	// ===============================================================
	//if (x == 5 && y == 1) {
	//	printf("%2.16g\n", charge);

	//printf("%g\n", source1);

	//}
	// ===============================================================
	// temporary variables (relaxation times)
	double tw0rp = omega_plus*dt;  //   omega_plus*dt 
	double tw0rm = omega_minus*dt; //   omega_minus*dt 
	double tw0cp = omega_c_plus*dt;  //   omega_c_plus*dt 
	double tw0cm = omega_c_minus*dt; //   omega_c_minus*dt 

	// TRT collision operations
	
	f0[gpu_field0_index(x, y, z)] = ft0 - (tw0rp * (fp0 - fep0) + tw0rm * (fm0 - fem0)) + dt*source0;
	h0[gpu_field0_index(x, y, z)] = ht0 - (tw0cp * (hp0 - hep0) + tw0cm * (hm0 - hem0));
	
	f2[gpu_fieldn_index(x, y, z, 1)] = ft1 - (tw0rp * (fp1 - fep1) + tw0rm * (fm1 - fem1)) + dt*source1;
	h2[gpu_fieldn_index(x, y, z, 1)] = ht1 - (tw0cp * (hp1 - hep1) + tw0cm * (hm1 - hem1));
	f2[gpu_fieldn_index(x, y, z, 2)] = ft2 - (tw0rp * (fp2 - fep2) + tw0rm * (fm2 - fem2)) + dt*source2;
	h2[gpu_fieldn_index(x, y, z, 2)] = ht2 - (tw0cp * (hp2 - hep2) + tw0cm * (hm2 - hem2));
	f2[gpu_fieldn_index(x, y, z, 3)] = ft3 - (tw0rp * (fp3 - fep3) + tw0rm * (fm3 - fem3)) + dt*source3;
	h2[gpu_fieldn_index(x, y, z, 3)] = ht3 - (tw0cp * (hp3 - hep3) + tw0cm * (hm3 - hem3));
	f2[gpu_fieldn_index(x, y, z, 4)] = ft4 - (tw0rp * (fp4 - fep4) + tw0rm * (fm4 - fem4)) + dt*source4;
	h2[gpu_fieldn_index(x, y, z, 4)] = ht4 - (tw0cp * (hp4 - hep4) + tw0cm * (hm4 - hem4));
	f2[gpu_fieldn_index(x, y, z, 5)] = ft5 - (tw0rp * (fp5 - fep5) + tw0rm * (fm5 - fem5)) + dt*source5;
	h2[gpu_fieldn_index(x, y, z, 5)] = ht5 - (tw0cp * (hp5 - hep5) + tw0cm * (hm5 - hem5));
	f2[gpu_fieldn_index(x, y, z, 6)] = ft6 - (tw0rp * (fp6 - fep6) + tw0rm * (fm6 - fem6)) + dt*source6;
	h2[gpu_fieldn_index(x, y, z, 6)] = ht6 - (tw0cp * (hp6 - hep6) + tw0cm * (hm6 - hem6));
	f2[gpu_fieldn_index(x, y, z, 7)] = ft7 - (tw0rp * (fp7 - fep7) + tw0rm * (fm7 - fem7)) + dt*source7;
	h2[gpu_fieldn_index(x, y, z, 7)] = ht7 - (tw0cp * (hp7 - hep7) + tw0cm * (hm7 - hem7));
	f2[gpu_fieldn_index(x, y, z, 8)] = ft8 - (tw0rp * (fp8 - fep8) + tw0rm * (fm8 - fem8)) + dt*source8;
	h2[gpu_fieldn_index(x, y, z, 8)] = ht8 - (tw0cp * (hp8 - hep8) + tw0cm * (hm8 - hem8));
	f2[gpu_fieldn_index(x, y, z, 9)] = ft9 - (tw0rp * (fp9 - fep9) + tw0rm * (fm9 - fem9)) + dt*source9;
	h2[gpu_fieldn_index(x, y, z, 9)] = ht9 - (tw0cp * (hp9 - hep9) + tw0cm * (hm9 - hem9));
	f2[gpu_fieldn_index(x, y, z, 10)] = ft10 - (tw0rp * (fp10 - fep10) + tw0rm * (fm10 - fem10)) + dt*source10;
	h2[gpu_fieldn_index(x, y, z, 10)] = ht10 - (tw0cp * (hp10 - hep10) + tw0cm * (hm10 - hem10));
	f2[gpu_fieldn_index(x, y, z, 11)] = ft11 - (tw0rp * (fp11 - fep11) + tw0rm * (fm11 - fem11)) + dt*source11;
	h2[gpu_fieldn_index(x, y, z, 11)] = ht11 - (tw0cp * (hp11 - hep11) + tw0cm * (hm11 - hem11));
	f2[gpu_fieldn_index(x, y, z, 12)] = ft12 - (tw0rp * (fp12 - fep12) + tw0rm * (fm12 - fem12)) + dt*source12;
	h2[gpu_fieldn_index(x, y, z, 12)] = ht12 - (tw0cp * (hp12 - hep12) + tw0cm * (hm12 - hem12));
	f2[gpu_fieldn_index(x, y, z, 13)] = ft13 - (tw0rp * (fp13 - fep13) + tw0rm * (fm13 - fem13)) + dt*source13;
	h2[gpu_fieldn_index(x, y, z, 13)] = ht13 - (tw0cp * (hp13 - hep13) + tw0cm * (hm13 - hem13));
	f2[gpu_fieldn_index(x, y, z, 14)] = ft14 - (tw0rp * (fp14 - fep14) + tw0rm * (fm14 - fem14)) + dt*source14;
	h2[gpu_fieldn_index(x, y, z, 14)] = ht14 - (tw0cp * (hp14 - hep14) + tw0cm * (hm14 - hem14));
	f2[gpu_fieldn_index(x, y, z, 15)] = ft15 - (tw0rp * (fp15 - fep15) + tw0rm * (fm15 - fem15)) + dt*source15;
	h2[gpu_fieldn_index(x, y, z, 15)] = ht15 - (tw0cp * (hp15 - hep15) + tw0cm * (hm15 - hem15));
	f2[gpu_fieldn_index(x, y, z, 16)] = ft16 - (tw0rp * (fp16 - fep16) + tw0rm * (fm16 - fem16)) + dt*source16;
	h2[gpu_fieldn_index(x, y, z, 16)] = ht16 - (tw0cp * (hp16 - hep16) + tw0cm * (hm16 - hem16));
	f2[gpu_fieldn_index(x, y, z, 17)] = ft17 - (tw0rp * (fp17 - fep17) + tw0rm * (fm17 - fem17)) + dt*source17;
	h2[gpu_fieldn_index(x, y, z, 17)] = ht17 - (tw0cp * (hp17 - hep17) + tw0cm * (hm17 - hem17));
	f2[gpu_fieldn_index(x, y, z, 18)] = ft18 - (tw0rp * (fp18 - fep18) + tw0rm * (fm18 - fem18)) + dt*source18;
	h2[gpu_fieldn_index(x, y, z, 18)] = ht18 - (tw0cp * (hp18 - hep18) + tw0cm * (hm18 - hem18));
	f2[gpu_fieldn_index(x, y, z, 19)] = ft19 - (tw0rp * (fp19 - fep19) + tw0rm * (fm19 - fem19)) + dt*source19;
	h2[gpu_fieldn_index(x, y, z, 19)] = ht19 - (tw0cp * (hp19 - hep19) + tw0cm * (hm19 - hem19));
	f2[gpu_fieldn_index(x, y, z, 20)] = ft20 - (tw0rp * (fp20 - fep20) + tw0rm * (fm20 - fem20)) + dt*source20;
	h2[gpu_fieldn_index(x, y, z, 20)] = ht20 - (tw0cp * (hp20 - hep20) + tw0cm * (hm20 - hem20));
	f2[gpu_fieldn_index(x, y, z, 21)] = ft21 - (tw0rp * (fp21 - fep21) + tw0rm * (fm21 - fem21)) + dt*source21;
	h2[gpu_fieldn_index(x, y, z, 21)] = ht21 - (tw0cp * (hp21 - hep21) + tw0cm * (hm21 - hem21));
	f2[gpu_fieldn_index(x, y, z, 22)] = ft22 - (tw0rp * (fp22 - fep22) + tw0rm * (fm22 - fem22)) + dt*source22;
	h2[gpu_fieldn_index(x, y, z, 22)] = ht22 - (tw0cp * (hp22 - hep22) + tw0cm * (hm22 - hem22));
	f2[gpu_fieldn_index(x, y, z, 23)] = ft23 - (tw0rp * (fp23 - fep23) + tw0rm * (fm23 - fem23)) + dt*source23;
	h2[gpu_fieldn_index(x, y, z, 23)] = ht23 - (tw0cp * (hp23 - hep23) + tw0cm * (hm23 - hem23));
	f2[gpu_fieldn_index(x, y, z, 24)] = ft24 - (tw0rp * (fp24 - fep24) + tw0rm * (fm24 - fem24)) + dt*source24;
	h2[gpu_fieldn_index(x, y, z, 24)] = ht24 - (tw0cp * (hp24 - hep24) + tw0cm * (hm24 - hem24));
	f2[gpu_fieldn_index(x, y, z, 25)] = ft25 - (tw0rp * (fp25 - fep25) + tw0rm * (fm25 - fem25)) + dt*source25;
	h2[gpu_fieldn_index(x, y, z, 25)] = ht25 - (tw0cp * (hp25 - hep25) + tw0cm * (hm25 - hem25));
	f2[gpu_fieldn_index(x, y, z, 26)] = ft26 - (tw0rp * (fp26 - fep26) + tw0rm * (fm26 - fem26)) + dt*source26;
	h2[gpu_fieldn_index(x, y, z, 26)] = ht26 - (tw0cp * (hp26 - hep26) + tw0cm * (hm26 - hem26));

}

__global__ void gpu_boundary(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2, double *f0bc)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;

	// set perturb = 0
	perturb = 0;

	// Full way bounce back
	if (z == 0) {
		// lower plate
		f0[gpu_field0_index(x, y, 0)]    = f0bc[gpu_field0_index(x, y, 0)];
		f2[gpu_fieldn_index(x, y, 0, 1)] = f1[gpu_fieldn_index(x, y, 0, 2)];
		f2[gpu_fieldn_index(x, y, 0, 2)] = f1[gpu_fieldn_index(x, y, 0, 1)];
		f2[gpu_fieldn_index(x, y, 0, 3)] = f1[gpu_fieldn_index(x, y, 0, 4)];
		f2[gpu_fieldn_index(x, y, 0, 4)] = f1[gpu_fieldn_index(x, y, 0, 3)];
		f2[gpu_fieldn_index(x, y, 0, 5)] = f1[gpu_fieldn_index(x, y, 0, 6)];
		f2[gpu_fieldn_index(x, y, 0, 6)] = f1[gpu_fieldn_index(x, y, 0, 5)];
		f2[gpu_fieldn_index(x, y, 0, 7)] = f1[gpu_fieldn_index(x, y, 0, 8)];
		f2[gpu_fieldn_index(x, y, 0, 8)] = f1[gpu_fieldn_index(x, y, 0, 7)];
		f2[gpu_fieldn_index(x, y, 0, 9)] = f1[gpu_fieldn_index(x, y, 0, 10)];
		f2[gpu_fieldn_index(x, y, 0, 10)] = f1[gpu_fieldn_index(x, y, 0, 9)];
		f2[gpu_fieldn_index(x, y, 0, 11)] = f1[gpu_fieldn_index(x, y, 0, 12)];
		f2[gpu_fieldn_index(x, y, 0, 12)] = f1[gpu_fieldn_index(x, y, 0, 11)];
		f2[gpu_fieldn_index(x, y, 0, 13)] = f1[gpu_fieldn_index(x, y, 0, 14)];
		f2[gpu_fieldn_index(x, y, 0, 14)] = f1[gpu_fieldn_index(x, y, 0, 13)];
		f2[gpu_fieldn_index(x, y, 0, 15)] = f1[gpu_fieldn_index(x, y, 0, 16)];
		f2[gpu_fieldn_index(x, y, 0, 16)] = f1[gpu_fieldn_index(x, y, 0, 15)];
		f2[gpu_fieldn_index(x, y, 0, 17)] = f1[gpu_fieldn_index(x, y, 0, 18)];
		f2[gpu_fieldn_index(x, y, 0, 18)] = f1[gpu_fieldn_index(x, y, 0, 17)];
		f2[gpu_fieldn_index(x, y, 0, 19)] = f1[gpu_fieldn_index(x, y, 0, 20)];
		f2[gpu_fieldn_index(x, y, 0, 20)] = f1[gpu_fieldn_index(x, y, 0, 19)];
		f2[gpu_fieldn_index(x, y, 0, 21)] = f1[gpu_fieldn_index(x, y, 0, 22)];
		f2[gpu_fieldn_index(x, y, 0, 22)] = f1[gpu_fieldn_index(x, y, 0, 21)];
		f2[gpu_fieldn_index(x, y, 0, 23)] = f1[gpu_fieldn_index(x, y, 0, 24)];
		f2[gpu_fieldn_index(x, y, 0, 24)] = f1[gpu_fieldn_index(x, y, 0, 23)];
		f2[gpu_fieldn_index(x, y, 0, 25)] = f1[gpu_fieldn_index(x, y, 0, 26)];
		f2[gpu_fieldn_index(x, y, 0, 26)] = f1[gpu_fieldn_index(x, y, 0, 25)];
		return;
	}

	// direction numbering scheme
	// 6 2 5
	// 3 0 1
	// 7 4 8
	// Boundary conditions
	double multis = 2.0*rho0*uw / cs_square * ws / CFL;
	double multia = 2.0*rho0*uw / cs_square * wa / CFL;
	double multid = 2.0*rho0*uw / cs_square * wd / CFL;
	if (z ==  NZ - 1) {
		// upper plate
		f0[gpu_field0_index(x, y, NZ - 1)]    = f0bc[gpu_field0_index(x, y, 1)];
		f2[gpu_fieldn_index(x, y, NZ - 1, 1)] = f1[gpu_fieldn_index(x, y, NZ - 1, 2)] + multis;
		f2[gpu_fieldn_index(x, y, NZ - 1, 2)] = f1[gpu_fieldn_index(x, y, NZ - 1, 1)] - multis;
		f2[gpu_fieldn_index(x, y, NZ - 1, 3)] = f1[gpu_fieldn_index(x, y, NZ - 1, 4)] + multis;
		f2[gpu_fieldn_index(x, y, NZ - 1, 4)] = f1[gpu_fieldn_index(x, y, NZ - 1, 3)];
		f2[gpu_fieldn_index(x, y, NZ - 1, 5)] = f1[gpu_fieldn_index(x, y, NZ - 1, 6)];
		f2[gpu_fieldn_index(x, y, NZ - 1, 6)] = f1[gpu_fieldn_index(x, y, NZ - 1, 5)];
		f2[gpu_fieldn_index(x, y, NZ - 1, 7)] = f1[gpu_fieldn_index(x, y, NZ - 1, 8)] + multia;
		f2[gpu_fieldn_index(x, y, NZ - 1, 8)] = f1[gpu_fieldn_index(x, y, NZ - 1, 7)] - multia;
		f2[gpu_fieldn_index(x, y, NZ - 1, 9)] = f1[gpu_fieldn_index(x, y, NZ - 1, 10)] + multia;
		f2[gpu_fieldn_index(x, y, NZ - 1, 10)] = f1[gpu_fieldn_index(x, y, NZ - 1, 9)] - multia;
		f2[gpu_fieldn_index(x, y, NZ - 1, 11)] = f1[gpu_fieldn_index(x, y, NZ - 1, 12)];
		f2[gpu_fieldn_index(x, y, NZ - 1, 12)] = f1[gpu_fieldn_index(x, y, NZ - 1, 11)];
		f2[gpu_fieldn_index(x, y, NZ - 1, 13)] = f1[gpu_fieldn_index(x, y, NZ - 1, 14)] + multia;
		f2[gpu_fieldn_index(x, y, NZ - 1, 14)] = f1[gpu_fieldn_index(x, y, NZ - 1, 13)] - multia;
		f2[gpu_fieldn_index(x, y, NZ - 1, 15)] = f1[gpu_fieldn_index(x, y, NZ - 1, 16)] + multia;
		f2[gpu_fieldn_index(x, y, NZ - 1, 16)] = f1[gpu_fieldn_index(x, y, NZ - 1, 15)] - multia;
		f2[gpu_fieldn_index(x, y, NZ - 1, 17)] = f1[gpu_fieldn_index(x, y, NZ - 1, 18)];
		f2[gpu_fieldn_index(x, y, NZ - 1, 18)] = f1[gpu_fieldn_index(x, y, NZ - 1, 17)];
		f2[gpu_fieldn_index(x, y, NZ - 1, 19)] = f1[gpu_fieldn_index(x, y, NZ - 1, 20)] + multid;
		f2[gpu_fieldn_index(x, y, NZ - 1, 20)] = f1[gpu_fieldn_index(x, y, NZ - 1, 19)] - multid;
		f2[gpu_fieldn_index(x, y, NZ - 1, 21)] = f1[gpu_fieldn_index(x, y, NZ - 1, 22)] + multid;
		f2[gpu_fieldn_index(x, y, NZ - 1, 22)] = f1[gpu_fieldn_index(x, y, NZ - 1, 21)] - multid;
		f2[gpu_fieldn_index(x, y, NZ - 1, 23)] = f1[gpu_fieldn_index(x, y, NZ - 1, 24)] + multid;
		f2[gpu_fieldn_index(x, y, NZ - 1, 24)] = f1[gpu_fieldn_index(x, y, NZ - 1, 23)] - multid;
		f2[gpu_fieldn_index(x, y, NZ - 1, 25)] = f1[gpu_fieldn_index(x, y, NZ - 1, 26)] - multid;
		f2[gpu_fieldn_index(x, y, NZ - 1, 26)] = f1[gpu_fieldn_index(x, y, NZ - 1, 25)] + multid;

		// Zero charge gradient on Ny
		h0[gpu_field0_index(x, y, NZ - 1)]    = h0[gpu_field0_index(x, y, NZ - 2)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 1)] = h2[gpu_fieldn_index(x, y, NZ - 2, 1)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 2)] = h2[gpu_fieldn_index(x, y, NZ - 2, 2)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 3)] = h2[gpu_fieldn_index(x, y, NZ - 2, 3)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 4)] = h2[gpu_fieldn_index(x, y, NZ - 2, 4)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 5)] = h2[gpu_fieldn_index(x, y, NZ - 2, 5)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 6)] = h2[gpu_fieldn_index(x, y, NZ - 2, 6)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 7)] = h2[gpu_fieldn_index(x, y, NZ - 2, 7)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 8)] = h2[gpu_fieldn_index(x, y, NZ - 2, 8)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 9)] = h2[gpu_fieldn_index(x, y, NZ - 2, 9)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 10)] = h2[gpu_fieldn_index(x, y, NZ - 2, 10)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 11)] = h2[gpu_fieldn_index(x, y, NZ - 2, 11)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 12)] = h2[gpu_fieldn_index(x, y, NZ - 2, 12)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 13)] = h2[gpu_fieldn_index(x, y, NZ - 2, 13)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 14)] = h2[gpu_fieldn_index(x, y, NZ - 2, 14)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 15)] = h2[gpu_fieldn_index(x, y, NZ - 2, 15)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 16)] = h2[gpu_fieldn_index(x, y, NZ - 2, 16)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 17)] = h2[gpu_fieldn_index(x, y, NZ - 2, 17)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 18)] = h2[gpu_fieldn_index(x, y, NZ - 2, 18)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 19)] = h2[gpu_fieldn_index(x, y, NZ - 2, 19)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 20)] = h2[gpu_fieldn_index(x, y, NZ - 2, 20)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 21)] = h2[gpu_fieldn_index(x, y, NZ - 2, 21)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 22)] = h2[gpu_fieldn_index(x, y, NZ - 2, 22)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 23)] = h2[gpu_fieldn_index(x, y, NZ - 2, 23)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 24)] = h2[gpu_fieldn_index(x, y, NZ - 2, 24)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 25)] = h2[gpu_fieldn_index(x, y, NZ - 2, 25)];
		h2[gpu_fieldn_index(x, y, NZ - 1, 26)] = h2[gpu_fieldn_index(x, y, NZ - 2, 26)];
		return;
	}
}

__global__ void gpu_stream(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2)
{
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	// streaming step
	unsigned int xp1 = (x + 1) % NX;
	unsigned int yp1 = (y + 1) % NY;
	unsigned int zp1 = (z + 1) % NZ;
	unsigned int xm1 = (NX + x - 1) % NX;
	unsigned int ym1 = (NY + y - 1) % NY;
	unsigned int zm1 = (NZ + z - 1) % NZ;
	// direction numbering scheme
	// 6 2 5
	// 3 0 1
	// 7 4 8

	// load populations from adjacent nodes (ft is post-streaming population of f1)
	// flows
	f1[gpu_fieldn_index(x, y, z, 1)] = f2[gpu_fieldn_index(xm1, y, z, 1)];
	f1[gpu_fieldn_index(x, y, z, 2)] = f2[gpu_fieldn_index(xp1, y, z, 2)];
	f1[gpu_fieldn_index(x, y, z, 3)] = f2[gpu_fieldn_index(x, ym1, z, 3)];
	f1[gpu_fieldn_index(x, y, z, 4)] = f2[gpu_fieldn_index(x, yp1, z, 4)];
	f1[gpu_fieldn_index(x, y, z, 5)] = f2[gpu_fieldn_index(x, y, zm1, 5)];
	f1[gpu_fieldn_index(x, y, z, 6)] = f2[gpu_fieldn_index(x, y, zp1, 6)];
	f1[gpu_fieldn_index(x, y, z, 7)] = f2[gpu_fieldn_index(xm1, ym1, z, 7)];
	f1[gpu_fieldn_index(x, y, z, 8)] = f2[gpu_fieldn_index(xp1, yp1, z, 8)];
	f1[gpu_fieldn_index(x, y, z, 9)] = f2[gpu_fieldn_index(xm1, y, zm1, 9)];
	f1[gpu_fieldn_index(x, y, z, 10)] = f2[gpu_fieldn_index(xp1, y, zp1, 10)];
	f1[gpu_fieldn_index(x, y, z, 11)] = f2[gpu_fieldn_index(x, ym1, zm1, 11)];
	f1[gpu_fieldn_index(x, y, z, 12)] = f2[gpu_fieldn_index(x, yp1, zp1, 12)];
	f1[gpu_fieldn_index(x, y, z, 13)] = f2[gpu_fieldn_index(xm1, yp1, z, 13)];
	f1[gpu_fieldn_index(x, y, z, 14)] = f2[gpu_fieldn_index(xp1, ym1, z, 14)];
	f1[gpu_fieldn_index(x, y, z, 15)] = f2[gpu_fieldn_index(xm1, y, zp1, 15)];
	f1[gpu_fieldn_index(x, y, z, 16)] = f2[gpu_fieldn_index(xp1, y, zm1, 16)];
	f1[gpu_fieldn_index(x, y, z, 17)] = f2[gpu_fieldn_index(x, ym1, zp1, 17)];
	f1[gpu_fieldn_index(x, y, z, 18)] = f2[gpu_fieldn_index(x, yp1, zm1, 18)];
	f1[gpu_fieldn_index(x, y, z, 19)] = f2[gpu_fieldn_index(xm1, ym1, zm1, 19)];
	f1[gpu_fieldn_index(x, y, z, 20)] = f2[gpu_fieldn_index(xp1, yp1, zp1, 20)];
	f1[gpu_fieldn_index(x, y, z, 21)] = f2[gpu_fieldn_index(xm1, ym1, zp1, 21)];
	f1[gpu_fieldn_index(x, y, z, 22)] = f2[gpu_fieldn_index(xp1, yp1, zm1, 22)];
	f1[gpu_fieldn_index(x, y, z, 23)] = f2[gpu_fieldn_index(xm1, yp1, zm1, 23)];
	f1[gpu_fieldn_index(x, y, z, 24)] = f2[gpu_fieldn_index(xp1, ym1, zp1, 24)];
	f1[gpu_fieldn_index(x, y, z, 25)] = f2[gpu_fieldn_index(xp1, ym1, zm1, 25)];
	f1[gpu_fieldn_index(x, y, z, 26)] = f2[gpu_fieldn_index(xm1, yp1, zp1, 26)];

	// charges
	h1[gpu_fieldn_index(x, y, z, 1)] = h2[gpu_fieldn_index(xm1, y, z, 1)];
	h1[gpu_fieldn_index(x, y, z, 2)] = h2[gpu_fieldn_index(xp1, y, z, 2)];
	h1[gpu_fieldn_index(x, y, z, 3)] = h2[gpu_fieldn_index(x, ym1, z, 3)];
	h1[gpu_fieldn_index(x, y, z, 4)] = h2[gpu_fieldn_index(x, yp1, z, 4)];
	h1[gpu_fieldn_index(x, y, z, 5)] = h2[gpu_fieldn_index(x, y, zm1, 5)];
	h1[gpu_fieldn_index(x, y, z, 6)] = h2[gpu_fieldn_index(x, y, zp1, 6)];
	h1[gpu_fieldn_index(x, y, z, 7)] = h2[gpu_fieldn_index(xm1, ym1, z, 7)];
	h1[gpu_fieldn_index(x, y, z, 8)] = h2[gpu_fieldn_index(xp1, yp1, z, 8)];
	h1[gpu_fieldn_index(x, y, z, 9)] = h2[gpu_fieldn_index(xm1, y, zm1, 9)];
	h1[gpu_fieldn_index(x, y, z, 10)] = h2[gpu_fieldn_index(xp1, y, zp1, 10)];
	h1[gpu_fieldn_index(x, y, z, 11)] = h2[gpu_fieldn_index(x, ym1, zm1, 11)];
	h1[gpu_fieldn_index(x, y, z, 12)] = h2[gpu_fieldn_index(x, yp1, zp1, 12)];
	h1[gpu_fieldn_index(x, y, z, 13)] = h2[gpu_fieldn_index(xm1, yp1, z, 13)];
	h1[gpu_fieldn_index(x, y, z, 14)] = h2[gpu_fieldn_index(xp1, ym1, z, 14)];
	h1[gpu_fieldn_index(x, y, z, 15)] = h2[gpu_fieldn_index(xm1, y, zp1, 15)];
	h1[gpu_fieldn_index(x, y, z, 16)] = h2[gpu_fieldn_index(xp1, y, zm1, 16)];
	h1[gpu_fieldn_index(x, y, z, 17)] = h2[gpu_fieldn_index(x, ym1, zp1, 17)];
	h1[gpu_fieldn_index(x, y, z, 18)] = h2[gpu_fieldn_index(x, yp1, zm1, 18)];
	h1[gpu_fieldn_index(x, y, z, 19)] = h2[gpu_fieldn_index(xm1, ym1, zm1, 19)];
	h1[gpu_fieldn_index(x, y, z, 20)] = h2[gpu_fieldn_index(xp1, yp1, zp1, 20)];
	h1[gpu_fieldn_index(x, y, z, 21)] = h2[gpu_fieldn_index(xm1, ym1, zp1, 21)];
	h1[gpu_fieldn_index(x, y, z, 22)] = h2[gpu_fieldn_index(xp1, yp1, zm1, 22)];
	h1[gpu_fieldn_index(x, y, z, 23)] = h2[gpu_fieldn_index(xm1, yp1, zm1, 23)];
	h1[gpu_fieldn_index(x, y, z, 24)] = h2[gpu_fieldn_index(xp1, ym1, zp1, 24)];
	h1[gpu_fieldn_index(x, y, z, 25)] = h2[gpu_fieldn_index(xp1, ym1, zm1, 25)];
	h1[gpu_fieldn_index(x, y, z, 26)] = h2[gpu_fieldn_index(xm1, yp1, zp1, 26)];
}

__global__ void gpu_bc_charge(double *h0, double *h1, double *h2)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y;
	unsigned int z = blockIdx.z;

	if (z == 0) {
		double multi0c = 2.0*charge0*w0;
		double multisc = 2.0*charge0*ws;
		double multiac = 2.0*charge0*wa;
		double multidc = 2.0*charge0*wd;
		// lower plate for charge density
		double ht1 = h2[gpu_fieldn_index(x, y, 0, 1)];
		double ht2 = h2[gpu_fieldn_index(x, y, 0, 2)];
		double ht3 = h2[gpu_fieldn_index(x, y, 0, 3)];
		double ht4 = h2[gpu_fieldn_index(x, y, 0, 4)];
		double ht5 = h2[gpu_fieldn_index(x, y, 0, 5)];
		double ht6 = h2[gpu_fieldn_index(x, y, 0, 6)];
		double ht7 = h2[gpu_fieldn_index(x, y, 0, 7)];
		double ht8 = h2[gpu_fieldn_index(x, y, 0, 8)];
		double ht9 = h2[gpu_fieldn_index(x, y, 0, 9)];
		double ht10 = h2[gpu_fieldn_index(x, y, 0, 10)];
		double ht11 = h2[gpu_fieldn_index(x, y, 0, 11)];
		double ht12 = h2[gpu_fieldn_index(x, y, 0, 12)];
		double ht13 = h2[gpu_fieldn_index(x, y, 0, 13)];
		double ht14 = h2[gpu_fieldn_index(x, y, 0, 14)];
		double ht15 = h2[gpu_fieldn_index(x, y, 0, 15)];
		double ht16 = h2[gpu_fieldn_index(x, y, 0, 16)];
		double ht17 = h2[gpu_fieldn_index(x, y, 0, 17)];
		double ht18 = h2[gpu_fieldn_index(x, y, 0, 18)];
		double ht19 = h2[gpu_fieldn_index(x, y, 0, 19)];
		double ht20 = h2[gpu_fieldn_index(x, y, 0, 20)];
		double ht21 = h2[gpu_fieldn_index(x, y, 0, 21)];
		double ht22 = h2[gpu_fieldn_index(x, y, 0, 22)];
		double ht23 = h2[gpu_fieldn_index(x, y, 0, 23)];
		double ht24 = h2[gpu_fieldn_index(x, y, 0, 24)];
		double ht25 = h2[gpu_fieldn_index(x, y, 0, 25)];
		double ht26 = h2[gpu_fieldn_index(x, y, 0, 26)];

		h0[gpu_field0_index(x, y, 0)] = -h0[gpu_field0_index(x, y, 0)] + multi0c;
		h1[gpu_fieldn_index(x, y, 0, 1)] = -ht2 + multisc;
		h1[gpu_fieldn_index(x, y, 0, 2)] = -ht1 + multisc;
		h1[gpu_fieldn_index(x, y, 0, 3)] = -ht4 + multisc;
		h1[gpu_fieldn_index(x, y, 0, 4)] = -ht3 + multisc;
		h1[gpu_fieldn_index(x, y, 0, 5)] = -ht6 + multisc;
		h1[gpu_fieldn_index(x, y, 0, 6)] = -ht5 + multisc;

		h1[gpu_fieldn_index(x, y, 0, 7)] = -ht8 + multiac;
		h1[gpu_fieldn_index(x, y, 0, 8)] = -ht7 + multiac;
		h1[gpu_fieldn_index(x, y, 0, 9)] = -ht10 + multiac;
		h1[gpu_fieldn_index(x, y, 0, 10)] = -ht9 + multiac;
		h1[gpu_fieldn_index(x, y, 0, 11)] = -ht12 + multiac;
		h1[gpu_fieldn_index(x, y, 0, 12)] = -ht11 + multiac;
		h1[gpu_fieldn_index(x, y, 0, 13)] = -ht14 + multiac;
		h1[gpu_fieldn_index(x, y, 0, 14)] = -ht13 + multiac;
		h1[gpu_fieldn_index(x, y, 0, 15)] = -ht16 + multiac;
		h1[gpu_fieldn_index(x, y, 0, 16)] = -ht15 + multiac;
		h1[gpu_fieldn_index(x, y, 0, 17)] = -ht18 + multiac;
		h1[gpu_fieldn_index(x, y, 0, 18)] = -ht17 + multiac;

		h1[gpu_fieldn_index(x, y, 0, 19)] = -ht20 + multidc;
		h1[gpu_fieldn_index(x, y, 0, 20)] = -ht19 + multidc;
		h1[gpu_fieldn_index(x, y, 0, 21)] = -ht22 + multidc;
		h1[gpu_fieldn_index(x, y, 0, 22)] = -ht21 + multidc;
		h1[gpu_fieldn_index(x, y, 0, 23)] = -ht24 + multidc;
		h1[gpu_fieldn_index(x, y, 0, 24)] = -ht23 + multidc;
		h1[gpu_fieldn_index(x, y, 0, 25)] = -ht26 + multidc;
		h1[gpu_fieldn_index(x, y, 0, 26)] = -ht25 + multidc;
		//======================================================================================================================================================================================
		//if (x == 0 && y == 0 && z == 0) test = h1[gpu_fieldn_index(x, y, z, 21)];// h1[gpu_fieldn_index(x, y, z, 1)];
		//======================================================================================================================================================================================
	}
}


__host__ void compute_parameters(double *T, double *M, double *C, double *Fe) {
	double K_host;
	double eps_host;
	double voltage_host;
	//double nu_host;
	double Ly_host;
	double diffu_host;
	double charge0_host;
	double rho0_host;

	cudaMemcpyFromSymbol(&K_host, K, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&eps_host, eps, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&voltage_host, voltage, sizeof(double), 0, cudaMemcpyDeviceToHost);
	//cudaMemcpyFromSymbol(&nu_host, nu, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&Ly_host, Ly, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&diffu_host, diffu, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&charge0_host, charge0, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&rho0_host, rho0, sizeof(double), 0, cudaMemcpyDeviceToHost);

	*M = sqrt(eps_host / rho0_host) / K_host;
	*T = eps_host*voltage_host / K_host / nu_host / rho0_host;
	*C = charge0_host * Lz_host * Lz_host / (voltage_host * eps_host);
	*Fe = K_host * voltage_host / diffu_host;
}

__host__ void report_flow_properties(unsigned int n, double t, double *rho, 
	double *charge, double *phi, double *ux, double *uy, double *uz, double *Ex, double *Ey, double *Ez)
{
    printf("Iteration: %u, physical time: %g.\n",n,t);
}

__host__ void save_scalar(double *scalar_gpu, double time)
{
	double *scalar_host = (double*)malloc(mem_size_scalar);

    // assume reasonably-sized file names
    char filename[128];
    char format[16];
    
    // compute maximum number of digits
    int ndigits = floor(log10((double)NSTEPS)+1.0);
    
    // generate format string
    // file name format is name0000nnn.bin
    sprintf(format,"%%s%%0%dd.bin",ndigits);
    sprintf(filename,"%g",time);
    
    // transfer memory from GPU to host
    checkCudaErrors(cudaMemcpy(scalar_host,scalar_gpu,mem_size_scalar,cudaMemcpyDeviceToHost));
    
    // open file for writing
    FILE *fout = fopen(filename,"wb+");
    
    // write data
    fwrite(scalar_host,1,mem_size_scalar,fout);
    
    // close file
    fclose(fout);
    
    if(ferror(fout))
    {
        fprintf(stderr,"Error saving to %s\n",filename);
        perror("");
    }
    else
    {
        if(!quiet)
            printf("Saved to %s\n",filename);
    }
}

__host__
void save_data_tecplot(FILE *fout, double time, double *rho_gpu, double *charge_gpu, double *phi_gpu,
	double *ux_gpu, double *uy_gpu, double *uz_gpu, double *Ex_gpu, double *Ey_gpu, double *Ez_gpu, int first) {
	
	double *rho    = (double*)malloc(mem_size_scalar);
	double *charge = (double*)malloc(mem_size_scalar);
	double *phi    = (double*)malloc(mem_size_scalar);
	double *ux     = (double*)malloc(mem_size_scalar);
	double *uy     = (double*)malloc(mem_size_scalar);
	double *uz     = (double*)malloc(mem_size_scalar);
	double *Ex     = (double*)malloc(mem_size_scalar);
	double *Ey     = (double*)malloc(mem_size_scalar);
	double *Ez     = (double*)malloc(mem_size_scalar);
	double dx_host;
	double dy_host;
	double dz_host;
	// transfer memory from GPU to host
	checkCudaErrors(cudaMemcpy(rho,    rho_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(charge, charge_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(phi,    phi_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(ux,     ux_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(uy,     uy_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(uz,     uz_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Ex,     Ex_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Ey,     Ey_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Ez,     Ez_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	cudaMemcpyFromSymbol(&dx_host, dx, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&dy_host, dy, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&dz_host, dz, sizeof(double), 0, cudaMemcpyDeviceToHost);	
	
	// apply boundary conditions (upper and lower plate)
	for (unsigned int y = 0; y < NY; ++y) {
		for (unsigned int x = 0; x < NX; ++x) {
			rho[scalar_index(x, y, 0)] = 2.0*rho[scalar_index(x, y, 1)] - rho[scalar_index(x, y, 2)];
			charge[scalar_index(x, y, 0)] = 2.0*charge[scalar_index(x, y, 1)] - charge[scalar_index(x, y, 2)];
			ux[scalar_index(x, y, 0)] = 2.0*ux[scalar_index(x, y, 1)] - ux[scalar_index(x, y, 2)];
			uy[scalar_index(x, y, 0)] = 2.0*uy[scalar_index(x, y, 1)] - uy[scalar_index(x, y, 2)];
			uz[scalar_index(x, y, 0)] = 2.0*uz[scalar_index(x, y, 1)] - uz[scalar_index(x, y, 2)];
			rho[scalar_index(x, y, NZ - 1)] = 2.0*rho[scalar_index(x, y, NZ - 2)] - rho[scalar_index(x, y, NZ - 3)];
			charge[scalar_index(x, y, NZ - 1)] = 2.0*charge[scalar_index(x, y, NZ - 2)] - charge[scalar_index(x, y, NZ - 3)];
			ux[scalar_index(x, y, NZ - 1)] = 2.0*ux[scalar_index(x, y, NZ - 2)] - ux[scalar_index(x, y, NZ - 3)];
			uy[scalar_index(x, y, NZ - 1)] = 2.0*uy[scalar_index(x, y, NZ - 2)] - uy[scalar_index(x, y, NZ - 3)];
			uz[scalar_index(x, y, NZ - 1)] = 2.0*uz[scalar_index(x, y, NZ - 2)] - uz[scalar_index(x, y, NZ - 3)];
		}
	}

	if (first)
	{
		char str[] = "VARIABLES=\"x\",\"y\",\"z\",\"u\",\"v\",\"w\",\"p\",\"charge\",\"phi\",\"Ex\",\"Ey\",\"Ez\"";
		//fwrite(str, 1, sizeof(str), fout);
		fprintf(fout, "%s\n", str);
	}
	fprintf(fout, "\n");
	fprintf(fout, "ZONE T=\"t=%g\", F=POINT, I = %d, J = %d, K = %d\n", time, NX, NY, NZ);

	for (unsigned int z = 0; z < NZ; ++z) 
	{
		for (unsigned int y = 0; y < NY; ++y)
		{
			for (unsigned int x = 0; x < NX; ++x)
			{
				fprintf(fout, "%g %g %g %g %g %g %g %g %10.6f %10.6f %10.6f %10.6f\n", dx_host*x, dy_host*y, dz_host*z,
					ux[scalar_index(x, y, z)], uy[scalar_index(x, y, z)], uz[scalar_index(x, y, z)], rho[scalar_index(x, y, z)], charge[scalar_index(x, y, z)],
					phi[scalar_index(x, y, z)], Ex[scalar_index(x, y, z)], Ey[scalar_index(x, y, z)], Ez[scalar_index(x, y, z)]);
			}
		}
	}
	free(rho);
	free(charge);
	free(phi);
	free(ux);
	free(uy);
	free(uz);
	free(Ex);
	free(Ey);
	free(Ez);
}

__host__
void save_data_end(FILE *fend, double time, double *rho_gpu, double *charge_gpu, double *phi_gpu,
	double *ux_gpu, double *uy_gpu, double *uz_gpu, double *Ex_gpu, double *Ey_gpu, double *Ez_gpu) {

	double *rho = (double*)malloc(mem_size_scalar);
	double *charge = (double*)malloc(mem_size_scalar);
	double *phi = (double*)malloc(mem_size_scalar);
	double *ux = (double*)malloc(mem_size_scalar);
	double *uy = (double*)malloc(mem_size_scalar);
	double *uz = (double*)malloc(mem_size_scalar);
	double *Ex = (double*)malloc(mem_size_scalar);
	double *Ey = (double*)malloc(mem_size_scalar);
	double *Ez = (double*)malloc(mem_size_scalar);

	// transfer memory from GPU to host
	checkCudaErrors(cudaMemcpy(rho, rho_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(charge, charge_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(phi, phi_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(ux, ux_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(uy, uy_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(uz, uz_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Ex, Ex_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Ey, Ey_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Ez, Ez_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));

	// apply boundary conditions (upper and lower plate)
	for (unsigned int y = 0; y < NY; ++y) {
		for (unsigned int x = 0; x < NX; ++x) {
			rho[scalar_index(x, y, 0)] = 2.0*rho[scalar_index(x, y, 1)] - rho[scalar_index(x, y, 2)];
			charge[scalar_index(x, y, 0)] = 2.0*charge[scalar_index(x, y, 1)] - charge[scalar_index(x, y, 2)];
			ux[scalar_index(x, y, 0)] = 2.0*ux[scalar_index(x, y, 1)] - ux[scalar_index(x, y, 2)];
			uy[scalar_index(x, y, 0)] = 2.0*uy[scalar_index(x, y, 1)] - uy[scalar_index(x, y, 2)];
			uz[scalar_index(x, y, 0)] = 2.0*uz[scalar_index(x, y, 1)] - uz[scalar_index(x, y, 2)];
			rho[scalar_index(x, y, NZ - 1)] = 2.0*rho[scalar_index(x, y, NZ - 2)] - rho[scalar_index(x, y, NZ - 3)];
			charge[scalar_index(x, y, NZ - 1)] = 2.0*charge[scalar_index(x, y, NZ - 2)] - charge[scalar_index(x, y, NZ - 3)];
			ux[scalar_index(x, y, NZ - 1)] = 2.0*ux[scalar_index(x, y, NZ - 2)] - ux[scalar_index(x, y, NZ - 3)];
			uy[scalar_index(x, y, NZ - 1)] = 2.0*uy[scalar_index(x, y, NZ - 2)] - uy[scalar_index(x, y, NZ - 3)];
			uz[scalar_index(x, y, NZ - 1)] = 2.0*uz[scalar_index(x, y, NZ - 2)] - uz[scalar_index(x, y, NZ - 3)];
		}
	}
	for (unsigned int z = 0; z < NZ; ++z)
	{
		for (unsigned int y = 0; y < NY; ++y)
		{
			for (unsigned int x = 0; x < NX; ++x)
			{
				fprintf(fend, "%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n", time,
					ux[scalar_index(x, y, z)], uy[scalar_index(x, y, z)], uz[scalar_index(x, y, z)], rho[scalar_index(x, y, z)], charge[scalar_index(x, y, z)],
					phi[scalar_index(x, y, z)], Ex[scalar_index(x, y, z)], Ey[scalar_index(x, y, z)], Ez[scalar_index(x, y, z)]);
			}
		}
	}
	free(rho);
	free(charge);
	free(phi);
	free(ux);
	free(uy);
	free(uz);
	free(Ex);
	free(Ey);
	free(Ez);
}

__host__
void save_data_dmd(FILE *fend, double time, double *uz_gpu) {

	double *uz = (double*)malloc(mem_size_scalar);

	// transfer memory from GPU to host
	checkCudaErrors(cudaMemcpy(uz, uz_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));

	// apply boundary conditions (upper and lower plate)
	for (unsigned int y = 0; y < NY; ++y) {
		for (unsigned int x = 0; x < NX; ++x) {
			uz[scalar_index(x, y, 0)] = 2.0*uz[scalar_index(x, y, 1)] - uz[scalar_index(x, y, 2)];
			uz[scalar_index(x, y, NZ - 1)] = 2.0*uz[scalar_index(x, y, NZ - 2)] - uz[scalar_index(x, y, NZ - 3)];
		}
	}
	for (unsigned int z = 0; z < NZ; ++z)
	{
		for (unsigned int y = 0; y < NY; ++y)
		{
			for (unsigned int x = 0; x < NX; ++x)
			{
				fprintf(fend, "%10.6f %10.6f\n", time, uz[scalar_index(x, y, z)]);
			}
		}
	}
	free(uz);
}

__host__
void read_data(double *time, double *rho_gpu, double *charge_gpu, double *phi_gpu,
	double *ux_gpu, double *uy_gpu, double *uz_gpu, double *Ex_gpu, double *Ey_gpu, double *Ez_gpu) {

	double *rho = (double*)malloc(mem_size_scalar);
	double *charge = (double*)malloc(mem_size_scalar);
	double *phi = (double*)malloc(mem_size_scalar);
	double *ux = (double*)malloc(mem_size_scalar);
	double *uy = (double*)malloc(mem_size_scalar);
	double *uz = (double*)malloc(mem_size_scalar);
	double *Ex = (double*)malloc(mem_size_scalar);
	double *Ey = (double*)malloc(mem_size_scalar);
	double *Ez = (double*)malloc(mem_size_scalar);

	FILE *fread = fopen("data_end.dat", "r");
	for (unsigned int z = 0; z < NZ; ++z)
	{
		for (unsigned int y = 0; y < NY; ++y)
		{
			for (unsigned int x = 0; x < NX; ++x)
			{
				fscanf(fread, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", time,
					&ux[scalar_index(x, y, z)], &uy[scalar_index(x, y, z)], &uz[scalar_index(x, y, z)], &rho[scalar_index(x, y, z)], &charge[scalar_index(x, y, z)],
					&phi[scalar_index(x, y, z)], &Ex[scalar_index(x, y, z)], &Ey[scalar_index(x, y, z)], &Ez[scalar_index(x, y, z)]);
			}
		}
	}
	// transfer memory from GPU to host
	checkCudaErrors(cudaMemcpy(rho_gpu, rho, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(charge_gpu, charge, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(phi_gpu, phi, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(ux_gpu, ux, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(uy_gpu, uy, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(uz_gpu, uz, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Ex_gpu, Ex, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Ey_gpu, Ey, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Ez_gpu, Ez, mem_size_scalar, cudaMemcpyHostToDevice));
	fclose(fread);
	free(rho);
	free(charge);
	free(phi);
	free(ux);
	free(uy);
	free(uz);
	free(Ex);
	free(Ey);
	free(Ez);
}


__host__
double current(double* c, double* ez){
	double I = 0;
	// apply boundary conditions (upper and lower plate)
	for (unsigned int y = 0; y < NY; ++y) {
		for (unsigned int x = 0; x < NX; ++x) {
			//rho[scalar_index(x, y, 0)] = 2.0*rho[scalar_index(x, y, 1)] - rho[scalar_index(x, y, 2)];
			c[scalar_index(x, y, 0)] = 2.0*c[scalar_index(x, y, 1)] - c[scalar_index(x, y, 2)];
			//ux[scalar_index(x, y, 0)] = 2.0*ux[scalar_index(x, y, 1)] - ux[scalar_index(x, y, 2)];
			//uy[scalar_index(x, y, 0)] = 2.0*uy[scalar_index(x, y, 1)] - uy[scalar_index(x, y, 2)];
			//uz[scalar_index(x, y, 0)] = 2.0*uz[scalar_index(x, y, 1)] - uz[scalar_index(x, y, 2)];
			//rho[scalar_index(x, y, NZ - 1)] = 2.0*rho[scalar_index(x, y, NZ - 2)] - rho[scalar_index(x, y, NZ - 3)];
			c[scalar_index(x, y, NZ - 1)] = 2.0*c[scalar_index(x, y, NZ - 2)] - c[scalar_index(x, y, NZ - 3)];
			//ux[scalar_index(x, y, NZ - 1)] = 2.0*ux[scalar_index(x, y, NZ - 2)] - ux[scalar_index(x, y, NZ - 3)];
			//uy[scalar_index(x, y, NZ - 1)] = 2.0*uy[scalar_index(x, y, NZ - 2)] - uy[scalar_index(x, y, NZ - 3)];
			//uz[scalar_index(x, y, NZ - 1)] = 2.0*uz[scalar_index(x, y, NZ - 2)] - uz[scalar_index(x, y, NZ - 3)];
		}
	}
	for(unsigned int y = 0; y<NY; y++){
		for(unsigned int x = 0; x<NX; x++){
			I += c[scalar_index(x,y,NZ-1)]*ez[scalar_index(x,y,NZ-1)];
		}
	}
	I = I * K_host * dy_host * dy_host;
	return I;
}

__host__
void record_umax(FILE *fend, double time, double *ux_gpu, double *uy_gpu, double *uz_gpu) {

	double *ux  = (double*)malloc(mem_size_scalar);
	double *uy  = (double*)malloc(mem_size_scalar);
	double *uz  = (double*)malloc(mem_size_scalar);
	double umax = 0;


	// transfer memory from GPU to host
	checkCudaErrors(cudaMemcpy(ux, ux_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(uy, uy_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(uz, uz_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));

	// apply boundary conditions (upper and lower plate)
	for (unsigned int y = 0; y < NY; ++y) {
		for (unsigned int x = 0; x < NX; ++x) {
			ux[scalar_index(x, y, NZ - 1)] = 2.0*ux[scalar_index(x, y, NZ - 2)] - ux[scalar_index(x, y, NZ - 3)];
			uy[scalar_index(x, y, NZ - 1)] = 2.0*uy[scalar_index(x, y, NZ - 2)] - uy[scalar_index(x, y, NZ - 3)];
			uz[scalar_index(x, y, NZ - 1)] = 2.0*uz[scalar_index(x, y, NZ - 2)] - uz[scalar_index(x, y, NZ - 3)];
		}
	}
	for (unsigned int z = 0; z < NZ; ++z)
	{
		for (unsigned int y = 0; y < NY; ++y)
		{
			for (unsigned int x = 0; x < NX; ++x)
			{
				//umax = MAX(umax, sqrt(ux[scalar_index(x, y, z)] * ux[scalar_index(x, y, z)] + uy[scalar_index(x, y, z)] * uy[scalar_index(x, y, z)]
				//	+ uz[scalar_index(x, y, z)] * uz[scalar_index(x, y, z)]));
				umax = MAX(umax,uz[scalar_index(x, y, z)]);
			}
		}
	}

	fprintf(fend, "%10.6f %10.6f\n", time, umax);

	free(ux);
	free(uy);
	free(uz);
}





