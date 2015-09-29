/****h* Bluebottle/bluebottle
 * NAME
 *  bluebottle
 * FUNCTION
 *  Bluebottle main execution code and global variable declarations.
 ******
 */

#ifndef _SCALAR_H
#define _SCALAR_H

#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/*
#ifdef DOUBLE
  typedef double real;
#else
  typedef float real;
#endif
*/
#include "domain.h"
#include "recorder.h"
#include "vtk.h"


#define MAX_THREADS_DIM3 10


//used in and lpt_integrate_mol and lpt_source_scalar_serial
#define LEN_GAUSSIAN_ARRAY 5000
#define KERNEL_WIDTH 1
#define STENCIL KERNEL_WIDTH*3
#define STENCIL2 STENCIL*STENCIL
#define STENCIL3 STENCIL2*STENCIL

#define EPSP_CLIP 0.8

extern int rec_scalar_stepnum_out;
extern real rec_scalar_field_dt;
//extern real rec_scalar_field;

//scalar field on host and device
extern real *sc;
extern real *sc0;
extern real **_sc;
extern real **_sc0;

//scalar diffusion, convective term on host and device
extern real *diff0_sc;
extern real **_diff0_sc;
extern real *conv0_sc;
extern real **_conv0_sc;
extern real *diff_sc;
extern real **_diff_sc;
extern real *conv_sc;
extern real **_conv_sc;

//yank boundary condition for scalar on host and device, haven't used yet
extern real *sc_WE;
extern real *sc_SN;
extern real *sc_BT;
extern real **_sc_WE;
extern real **_sc_SN;
extern real **_sc_BT;
extern real DIFF;//Diffusion coefficient for oil drop-ambient flow mass transfer
extern real DIFF_eq;//Diffusion coefficient for scalar diffusion equation
extern real DIFF_dt;//scalar source diffusion time scale

//source for scalar diffusion, on host on device
extern real *scSrc;
extern real **_scSrc;

//particle volume fraction in each grid cell, on host on device
extern real *epsp;
extern real **_epsp;

//time step for scalar diffusion
extern real dt_sc;

//if time step is scalar is smaller than for flow field, do sub-steps for scalar during a velocity iteration time step
//dt_try=min(dt_sc,dt-dt_done)
extern real dt_try;
extern real dt0_try;

//dt_done =sum(dt_try), if dt_done=dt, the sub-steps for scalar are finished
extern real dt_done;
//ttime_done=dt_done+ttime, real time during the sub-steps
extern real ttime_done;

//initial condition of the scalar field, default is QUIESCENT
extern int sc_init_cond;
extern real sc_aver;

//Equlibrium concentration of scalar at the interface between flow field and particles
extern real sc_eq;

//Initial value of the scalar field given the initial condition for scalar field is QUIESCENT
extern real sc_init_value;


//Determins whether soluble mass in particle will diffuse into the scalar field
//if sc_twoway>0 it will influence scalar field, otherwise it will not.
extern real sc_twoway;

//special parameters to determine the velocity scale and time scale of Cellular flow when calculating oil diffusion
extern real n2;
extern real n3;

//For yank of initial conditions of scalar field.  Not used yet
extern real bc_sc_configs[18];
extern real bc_sc_vals[18];


//Allocate device memory for scalar field
void cuda_scalar_malloc(void);

//read input from scalar.config
void scalar_read_input(void);

//show scalar.config in debug mode
void scalar_show_config(void);

//initilize scalar field based on the input
int scalar_init(void);

//Free host memories for scalar
void scalar_clean(void);

//Advance the scalar diffusion equation
void cuda_scalar_advance(void);

//Find the maximum time-step for scalar field, dt is the time-step of flow field for comparison
real cuda_find_dt_sc(real dt);
//Find the maximum time-step allowed to make sure that no scalar goes beyond equlibrium value
real cuda_find_dt_source(real dt);

void cuda_diffScalar_sub_explicitH(int coordiSys,int dev, real *scSrc);





/****s* bluebottle/BC
 * NAME
 *  BC
 * TYPE
 */
typedef struct scBC {
  int scW;
  real scWDm;
  real scWD;
  real scWDa;
  int scE;
  real scEDm;
  real scED;
  real scEDa;
  int scS;
  real scSDm;
  real scSD;
  real scSDa;
  int scN;
  real scNDm;
  real scND;
  real scNDa;
  int scB;
  real scBDm;
  real scBD;
  real scBDa;
  int scT;
  real scTDm;
  real scTD;
  real scTDa;

} scBC;
/*
 * PURPOSE
 *  Carry the type of boundary condition on each side of the domain.  Possible
 *  types include:
 *  * PERIODIC
 *  * DIRICHLET
 *  * NEUMANN
 *  * PRECURSOR
 *  If the boundary type is DIRICHLET or PRECURSOR, the value of the field
 *  variable on the boundary must be defined.
 * MEMBERS
 *  * scW -- the boundary condition type
 *  * scWDm -- the maximum DIRICHLET boundary condition value
 *  * scWD -- the current DIRICHLET boundary conditon value
 *  * scWDa -- the DIRICHLET boundary condition value acceleration
 *  * scE -- the boundary condition type
 *  * scEDm -- the maximum DIRICHLET boundary condition value
 *  * scED -- the current DIRICHLET boundary conditon value
 *  * scEDa -- the DIRICHLET boundary condition value acceleration
 *  * scS -- the boundary condition type
 *  * scSDm -- the maximum DIRICHLET boundary condition value
 *  * scSD -- the current DIRICHLET boundary conditon value
 *  * scSDa -- the DIRICHLET boundary condition value acceleration
 *  * scN -- the boundary condition type
 *  * scND -- the maximum DIRICHLET boundary condition valuem
 *  * scND -- the current DIRICHLET boundary conditon value
 *  * scNDa -- the DIRICHLET boundary condition value acceleration
 *  * scB -- the boundary condition type
 *  * scBDm -- the maximum DIRICHLET boundary condition value
 *  * scBD -- the current DIRICHLET boundary conditon value
 *  * scBDa -- the DIRICHLET boundary condition value acceleration
 *  * scT -- the boundary condition type
 *  * scTDm -- the maximum DIRICHLET boundary condition value
 *  * scTD -- the current DIRICHLET boundary conditon value
 *  * scTDa -- the DIRICHLET boundary condition value acceleration
  ******
 */

/****v* bluebottle/bc
 * NAME
 *  bc
 * TYPE
 */
extern scBC sc_bc;
/*
 * PURPOSE
 *  Create an instance of the struct BC to carry boundary condition types.
 ******
 */





/****f* bluebottle/cuda_dom_push()
 * NAME
 *  cuda_dom_push()
 * USAGE
 */
void cuda_scalar_push(void);
/*
 * FUNCTION
 *  Copy sc, conv_sc,diff_sc from host to device.
 ******
 */





/****f* bluebottle/cuda_dom_pull()
 * NAME
 *  cuda_dom_pull()
 * USAGE
 */
void cuda_scalar_pull(void);
/*
 * FUNCTION
 *  Copy sc, conv_sc,diff_sc from device to host.
 ******
 */





/****f* bluebottle/cuda_dom_free()
 * NAME
 *  cuda_dom_free()
 * USAGE
 */
void cuda_scalar_free(void);
/*
 * FUNCTION
 *  Free device memory for the domain on device and device memory reference
 *  pointers on host.
 ******
 */


/****f* bluebottle/cuda_dom_BC()
 * NAME
 *  cuda_dom_BC()
 * USAGE
 */
void cuda_scalar_BC(void);
/*
 * FUNCTION
 *  Enforce boundary conditions in scalar fields.  *NOTE:*
 *  cuda_scalar_BC() implements PERIODIC boundary conditions for single-GPU only.
 *  Dirichlet and Neumann boundary conditions are supported on multi-GPU
 *  domain decompositions.
 ******
 */

/****f* domain/compute_scalar_BC()
 * NAME
 *  compute_scalar_BC()
 * USAGE
 */
void compute_scalar_BC(void);
/*
 * FUNCTION
 *  Set up the Dirichlet scalar boundary conditions for this time step
 *  given user-inputted maximum scalar and acceleration.
 ******
 */

/****f* bluebottle/cuda_store_scalar()
 * NAME
 *  cuda_store_scalar()
 * USAGE
 */
void cuda_store_scalar(void);
/*
 * FUNCTION
 *  Store the previous sc,conv_sc,diff_sc components for use in the next timestep.
 ******
 */



#endif
