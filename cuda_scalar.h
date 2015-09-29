/****h* Bluebottle/cuda_bluebottle_kernel
 * NAME
 *  cuda_bluebottle_kernel
 * FUNCTION
 *  Bluebottle CUDA kernel functions.
 ******
 */

#ifndef _CUDA_SCALAR_H
#define _CUDA_SCALAR_H

extern "C"
{
#include "bluebottle.h"
#include "scalar.h"
#include "point.h"
}

extern texture<float,1,cudaReadModeElementType> texRefGaussian;
extern texture<int,1,cudaReadModeElementType> texRefDomInfo;

__global__ void diffScalar_explicitD(real *sc,real *sc0,
				dom_struct *dom, real DIFF_dt);


/****f* cuda_bluebottle_kernel/BC_sc_W_P<<<>>>()
 * NAME
 *  BC_sc_W_P<<<>>>()
 * USAGE
 */
__global__ void BC_sc_W_P(real *p, dom_struct *dom);
/*
 * FUNCTION
 *  Apply periodic boundary conditions to the west face scalar field.
 * ARGUMENTS
 *  * p -- the device scalar field subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_sc_W_N<<<>>>()
 * NAME
 *  BC_sc_W_N<<<>>>()
 * USAGE
 */
__global__ void BC_sc_W_N(real *p, dom_struct *dom);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the west face scalar field.
 * ARGUMENTS
 *  * p -- the device scalar field subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_sc_E_P<<<>>>()
 * NAME
 *  BC_sc_E_P<<<>>>()
 * USAGE
 */
__global__ void BC_sc_E_P(real *p, dom_struct *dom);
/*
 * FUNCTION
 *  Apply periodic boundary conditions to the east face scalar field.
 * ARGUMENTS
 *  * p -- the device scalar field subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_sc_E_N<<<>>>()
 * NAME
 *  BC_sc_E_N<<<>>>()
 * USAGE
 */
__global__ void BC_sc_E_N(real *p, dom_struct *dom);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the east face scalar field.
 * ARGUMENTS
 *  * p -- the device scalar field subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_sc_S_P<<<>>>()
 * NAME
 *  BC_sc_S_P<<<>>>()
 * USAGE
 */
__global__ void BC_sc_S_P(real *p, dom_struct *dom);
/*
 * FUNCTION
 *  Apply periodic boundary conditions to the south face scalar field.
 * ARGUMENTS
 *  * p -- the device scalar field subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_sc_S_N<<<>>>()
 * NAME
 *  BC_sc_S_N<<<>>>()
 * USAGE
 */
__global__ void BC_sc_S_N(real *p, dom_struct *dom);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the south face scalar field.
 * ARGUMENTS
 *  * p -- the device scalar field subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_sc_N_P<<<>>>()
 * NAME
 *  BC_sc_N_P<<<>>>()
 * USAGE
 */
__global__ void BC_sc_N_P(real *p, dom_struct *dom);
/*
 * FUNCTION
 *  Apply periodic boundary conditions to the north face scalar field.
 * ARGUMENTS
 *  * p -- the device scalar field subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_sc_N_N<<<>>>()
 * NAME
 *  BC_sc_N_N<<<>>>()
 * USAGE
 */
__global__ void BC_sc_N_N(real *p, dom_struct *dom);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the north face scalar field.
 * ARGUMENTS
 *  * p -- the device scalar field subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_sc_B_P<<<>>>()
 * NAME
 *  BC_sc_B_P<<<>>>()
 * USAGE
 */
__global__ void BC_sc_B_P(real *p, dom_struct *dom);
/*
 * FUNCTION
 *  Apply periodic boundary conditions to the bottom face scalar field.
 * ARGUMENTS
 *  * p -- the device scalar field subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_sc_B_N<<<>>>()
 * NAME
 *  BC_sc_B_N<<<>>>()
 * USAGE
 */
__global__ void BC_sc_B_N(real *p, dom_struct *dom);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the bottom face scalar field.
 * ARGUMENTS
 *  * p -- the device scalar field subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_sc_T_P<<<>>>()
 * NAME
 *  BC_sc_T_P<<<>>>()
 * USAGE
 */
__global__ void BC_sc_T_P(real *p, dom_struct *dom);
/*
 * FUNCTION
 *  Apply periodic boundary conditions to the top face scalar field.
 * ARGUMENTS
 *  * p -- the device scalar field subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_bluebottle_kernel/BC_sc_T_N<<<>>>()
 * NAME
 *  BC_sc_T_N<<<>>>()
 * USAGE
 */
__global__ void BC_sc_T_N(real *p, dom_struct *dom);
/*
 * FUNCTION
 *  Apply Neumann boundary conditions to the top face scalar field.
 * ARGUMENTS
 *  * p -- the device scalar field subdomain
 *  * dom -- the device subdomain
 ******
 */


/****f* cuda_bluebottle_kernel/BC_sc_W_D<<<>>>()
 * NAME
 *  BC_sc_W_D<<<>>>()
 * USAGE
 */
__global__ void BC_sc_W_D(real *sc, dom_struct *dom, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the west face sc-scalar field.
 * ARGUMENTS
 *  * sc -- the device sc-scalar field subdomain
 *  * dom -- the device subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */



/****f* cuda_bluebottle_kernel/BC_sc_E_D<<<>>>()
 * NAME
 *  BC_sc_E_D<<<>>>()
 * USAGE
 */
__global__ void BC_sc_E_D(real *sc, dom_struct *dom, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the east face sc-scalar field.
 * ARGUMENTS
 *  * sc -- the device sc-scalar field subdomain
 *  * dom -- the device subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */


/****f* cuda_bluebottle_kernel/BC_sc_S_D<<<>>>()
 * NAME
 *  BC_sc_S_D<<<>>>()
 * USAGE
 */
__global__ void BC_sc_S_D(real *sc, dom_struct *dom, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the south face sc-scalar field.
 * ARGUMENTS
 *  * sc -- the device sc-scalar field subdomain
 *  * dom -- the device subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */


/****f* cuda_bluebottle_kernel/BC_sc_N_D<<<>>>()
 * NAME
 *  BC_sc_N_D<<<>>>()
 * USAGE
 */
__global__ void BC_sc_N_D(real *sc, dom_struct *dom, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the north face sc-scalar field.
 * ARGUMENTS
 *  * sc -- the device sc-scalar field subdomain
 *  * dom -- the device subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */



/****f* cuda_bluebottle_kernel/BC_sc_T_D<<<>>>()
 * NAME
 *  BC_sc_T_D<<<>>>()
 * USAGE
 */
__global__ void BC_sc_T_D(real *sc, dom_struct *dom, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the top face sc-scalar field.
 * ARGUMENTS
 *  * sc -- the device sc-scalar field subdomain
 *  * dom -- the device subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */


/****f* cuda_bluebottle_kernel/BC_sc_B_D<<<>>>()
 * NAME
 *  BC_sc_B_D<<<>>>()
 * USAGE
 */
__global__ void BC_sc_B_D(real *sc, dom_struct *dom, real bc);
/*
 * FUNCTION
 *  Apply Dirichlet boundary conditions to the bottom face sc-scalar field.
 * ARGUMENTS
 *  * sc -- the device sc-scalar field subdomain
 *  * dom -- the device subdomain
 *  * bc -- the value to be set on the boundary
 ******
 */





__global__ void advance_sc_upwind_1st_init(real DIFF, real *u, real *v, real *w,real *f,real *epsp,
  real *diff0, real *conv0, real *diff, real *conv, real *sc,real *sc0,
  dom_struct *dom, real dt0, real dt);


//first order upwind
__global__ void advance_sc_upwind_1st(real DIFF, real *u, real *v, real *w,real *f,real *epsp,
  real *diff0, real *conv0, real *diff, real *conv, real *sc,real *sc0,
  dom_struct *dom, real dt0, real dt);






/*
Advance the scalar based on fluid velocity and particle volume fraction, using Adam-bashforth method in time and
central difference in space
	DIFF is scalar diffusion coeffcient,
	u~v are the face-center velocity of flow, f is the body force on flow
	epsp is the  particle volume fraction at each grid cell
	diff,conv are the diffusive and convective term of the scalar.  diff0,conv0 are for old step
	sc is the scalar field, sc0 for old step. 
	dt,dt0 are time step of scalar diffusion equation
*/
__global__ void advance_sc(real DIFF, real *u, real *v, real *w,real *f,real *epsp,
  real *diff0, real *conv0, real *diff, real *conv,real *sc,real *sc0,
  dom_struct *dom, real dt0, real dt);

//at the very 1st step to advance scalar without Adam-bashforth interpolation
__global__ void advance_sc_init(real DIFF, real *u, real *v, real *w,real *f,real *epsp,
  real *diff0, real *conv0, real *diff, real *conv,real *sc,real *sc0,
  dom_struct *dom, real dt0, real dt);

//Advance scalar using MacCormack scheme.
__global__ void advance_sc_macCormack(	real DIFF, 
				real *u, real *v, real *w,
				real *f,real *epsp,
				real *diff, real *conv, 
				real *sc,real *sc0,
				dom_struct *dom, real dt);

__global__ void advance_sc_QUICK(real DIFF, 
				real *u, real *v, real *w,
				real *f,real *epsp,
				real *diff, real *conv, 
				real *sc,real *sc0,
				dom_struct *dom, real dt);

//initialize flow field array to be 0 on device, include scalar source and particle volume fraction, array length is dom->s3b
__global__ void lpt_scalar_source_init(real *A, real *B,dom_struct *dom);

//initialize particle property array on device, include volume of point particle and source contributed by point particle, array length is equal to particle number
__global__ void   lpt_point_source_init(int npoints,point_struct *points,dom_struct *dom,real *volPoint,real *scSrc);

//clip volume fraction of particle in each grid cell, if it's bigger than a threshold(default:0.8), then clip it
__global__ void lpt_epsp_clip(real *epsp,dom_struct *dom);


//lpt_source_test
__global__ void lpt_scalar_source_init_test(real *src,dom_struct *dom,real t, real DIFF);

__global__ void lpt_scalar_source_convDiff_test(real *src,dom_struct *dom,real t, real DIFF);

#endif
