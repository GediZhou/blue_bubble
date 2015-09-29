/****h* Bluebottle/cuda_point_point_particle_kernel
 * NAME
 *  cuda_bluebottle_kernel
 * FUNCTION
 *  Bluebottle CUDA point_point_particle kernel functions.
 ******
 */

#ifndef _CUDA_POINT_H
#define _CUDA_POINT_H

extern "C"
{
#include "bluebottle.h"
//#include "point.h"
}

extern texture<float,1,cudaReadModeElementType> texRefGaussian;
extern texture<int,1,cudaReadModeElementType> texRefDomInfo;

void cuda_free_array_int(int** &A);
void cuda_free_array_real(real**& A);
void cuda_malloc_array_int(int**& A,int len);
void cuda_malloc_array_real(real**& A,int len);
real cuda_find_max_dia(void);

//void match_point_vel_with_flow(void);

void lpt_point_source_mollify_init();
void gaussian_array_initH();
void domInfo_array_initH();
void lpt_point_source_mollify_final();

//extern "C"
//real cuda_find_dt_points(real dt);

//mathch point velocity with the flow interpolated velocity at the particle position
__global__ void point_vel_specify(real *ug,real *vg,real *wg,point_struct *points,int npoints);


//initialize array A with length n, initialize the array to be a constant C 
__global__ void array_init(real *A,dom_struct *dom, int n, real C);

__global__ void print_kernel_array_int(int *cell,int lenCell);
__global__ void print_kernel_array_real(real *cell,int lenCell);

__global__ void copy_points_dt(real *pdt,point_struct *points,int npoints);

void sortParticles(int *dGridParticleHash, int *dGridParticleIndex, int npoints);

__global__ void entrySearch_max_int_kernel(int *g_iarr, int *g_max_intarr,
  int size);
__global__ void points_dp(point_struct *points, real *dp, int npoints);

__global__ void scan_serial(int *A, int *B, int size);


/*
The following two subroutines are for mollifying point source with a Gaussian kernel
	A is located at cell center, Ksi is the mollify coefficients, 
	coordiSys indicate whether mollify on cell center or face center. coordiSys=0 is cell center, coordiSys =1~3 are x-face to z-face center.  When on face center, the subroutine mollifies momentum source
	valType is for cell center only, indicate what property to mollify.  valType=0 is for mollification of scalar source, valType=1 is  for particle volume fraction 'epsp '
*/

//__global__ void lpt_mollify_sc(int npoints, real *A, point_struct *points, dom_struct *dom,real *Ksi, int coordiSys, int valType);
//__global__ void lpt_sum_ksi(int npoints, real *A, point_struct *points, dom_struct *dom,real *Ksi, int coordiSys,int valType);

int find_max_int(int size, int *d_iarr);
extern "C"
void lpt_mollify_scH(int coordiSys,int valType,int dev,real *scSrc);

extern "C"
void lpt_mollify_sc_optH(int coordiSys,int valType,int dev,real *scSrc);

extern "C"
void lpt_mollify_delta_scH(int coordiSys,int valType,int dev,real *scSrc);

__device__ void calcGridPos_opt(int &ip,int &jp,int &kp,real xp,real yp,real zp,dom_struct *dom,int coordiSys);

//Use texture memory to fetch it!!!
//Gausian kernel to calculate weight coefficient of the filter
__device__ real lpt_integrate_mol_opt(int ic,int jc,int kc,real xp,real yp,real zp, dom_struct *dom, int coordiSys);


__global__ void gaussian_array_initD(float * GaussianKernel,real dg2_sig2,real dg,real norm);

__global__ void lpt_point_position(point_struct *points,real *posX,real *posY,real *posZ, int npoints);


__global__ void lpt_delta_point_position(point_struct *points,
dom_struct *dom,
real *posX,real *posY,real *posZ,
real *lptSourceVal, 
int npoints, int coordiSys,int valType,real ratio);


__global__ void findCellStart_deltaD(int   *cellStart,        // output: cell start pp
                                  int   *cellEnd,          // output: cell end pp
                                  int   *gridParticleHash, // input: sorted grid hashes
                                  int   *gridParticleIndex,   // input: sorted particle indices
				  real *lptSourceVal,
				  real *lptSourceValOld,
                                  int    npoints);


__global__
void lpt_mollify_delta_scD(dom_struct *dom,
              real *A,
	      real *lptSourceVal,
              int   *cellStart,
              int   *cellEnd,
              int    npoints,
	      int    coordiSys);


//This method will work with higher efficiency for thousands of particle
__global__
void lpt_mollify_sc_ksi_optD(dom_struct *dom,
              real *A,
	      real *posX,
	      real *posY,
	      real *posZ,
	      real *Ksi,
              int   *cellStart,
              int   *cellEnd,
              int   *gridParticleIndex,
              int    npoints,
              int coordiSys,int valType);

__global__
void lpt_mollify_sc_optD(dom_struct *dom,
              real *A,
	      real *posX,
	      real *posY,
	      real *posZ,
	      real *Weight,
              int   *cellStart,
              int   *cellEnd,
              int   *gridParticleIndex,
              int    npoints,
              int coordiSys,int valType);

__global__ void calcHash_optD(int   *gridParticleHash,  // output
               int   *gridParticleIndex, // output
               dom_struct *dom,               // input: dom info
	       real *posX,real *posY,real *posZ,
               int    npoints,
	       int coordiSys);

__global__ void lpt_point_weight(point_struct *points,
				 dom_struct *dom,
				 real *posX,real *posY,real *posZ, 
				 real *Weight,
				 int   *gridParticleIndex,
				 int npoints,int coordiSys, int valType,real ratio);

__global__ void findCellStart_optD(   int   *cellStart,        // output: cell start pp
                                  int   *cellEnd,          // output: cell end pp
                                  int   *gridParticleHash, // input: sorted grid hashes
                                  int   *gridParticleIndex,   // input: sorted particle indices
				  real *posX,real *posY,real *posZ,
				  real *posXold,real *posYold,real *posZold,
                                  int    npoints);
__global__
void calcGridFlowHash_optD(dom_struct *dom,
              int *gridFlowHash,
              int coordiSys);

//calculate the maximum number of particles in a single grid cell
__global__ void calcMaxPointsPerCell_optD(dom_struct *dom, int *cellStart, int *cellEnd,int *pointsNum, int coordiSys);




__global__ void lpt_point_ksi_opt(point_struct *points,
                                 dom_struct *dom,
                                 real *posX,real *posY,real *posZ,
                                 real *Ksi,
                                 int   *gridParticleIndex,
                                 int npoints,int coordiSys, int valType,real ratio);

//__device__ void calcGridPos(int &ip,int &jp,int &kp,real xp,real yp,real zp,int coordiSys,real xs,real ys,real zs,real ddx,real ddy,real ddz);
__device__ void calcGridPos(point_struct *points,dom_struct *dom,int pp,int coordiSys);
//__device__ int calcGridHash(int &ic, int &jc,int &kc,dom_struct *dom,int coordiSys);
__device__ int calcGridHash(int ic, int jc,int kc,dom_struct *dom,int coordiSys);

__device__ int mod_int(int x,int N);

__device__ void dom_startEnd_index(int &is, int &js,int &ks,int &ie, int &je,int &ke,dom_struct *dom,int coordiSys,int incGhost);
__device__ real lpt_mol_typeVal(point_struct *points,dom_struct *dom,int pp,int coordiSys,int valType,real ratio);



//Make the position (x,y,z) periodic
__device__ void periodic_grid_position(real &x,real &y,real &z,dom_struct *dom);


//Make the index (i,j,k) periodic
__device__ void periodic_grid_index(int &ic,int &jc,int &kc,dom_struct *dom, int coordiSys);

/*
Gaussian kernel with object position index (ic,jc,kc) and Gaussian center position (xp,yp,zp), basically calculate their distance substitute into Gaussian kernel. 
	dx~dz are grid cell size, xs~zs are start position of the whole fluid domain
	coordiSys =0 indicate cell-center; coordiSys =1,2,3 indicate x,y,z-face-center;
*/
//__device__ real lpt_integrate_mol(int ic,int jc,int kc,real xp,real yp,real zp,int coordiSys,real xs,real ys,real zs,real dx,real dy,real dz);
__device__ real lpt_integrate_mol(int ic,int jc,int kc,real xp,real yp,real zp, dom_struct *dom, int coordiSys);

__device__ real point_cell_ksi(int gridHash,
                   int is, int js, int ks,
                   real *Ksi,
                   int   *cellStart,
                   int   *cellEnd,
                   int   *gridParticleIndex);

__device__ real sum_ksi_cell( int ic,int jc,int kc,
                   point_struct *points,
                   dom_struct *dom,
		   real *Ksi,
                   int   *cellStart,
                   int   *cellEnd,
                   int   *gridParticleIndex,
		   int coordiSys,int valType);
__global__
void calcHashD(int   *gridParticleHash,  // output
               int   *gridParticleIndex, // output
               point_struct *points,               // input: particles
               dom_struct *dom,               // input: dom info
               int    npoints,
	       int coordiSys);
__global__ void lpt_mollify_scD( point_struct *points,
              dom_struct *dom,
	      real *A,
	      real *Ksi,
              int   *cellStart,
              int   *cellEnd,
	      int   *gridParticleIndex,
              int    npoints,
	      int coordiSys,int valType);

__global__ void findCellStartD(   int   *cellStart,        // output: cell start pp
                                  int   *cellEnd,          // output: cell end pp
                                  int   *gridParticleHash, // input: sorted grid hashes
                                  int   *gridParticleIndex,   // input: sorted particle indices
                                  int    npoints);

__global__ void lpt_point_ksi(   point_struct *points,
				 dom_struct *dom,
				 real *Ksi,
				 int  *gridParticleIndex,
				 int npoints,
				 int coordiSys, int valType);

//initialize the the velocity ug~wg and stress of fluid lpt_stress_u~w, which will be interpreted at the point particle position. All arrays will be initialized to 0
__global__ void point_interp_init(int npoints,point_struct *points,real *ug,real *vg,real *wg,real *lpt_stress_u,real *lpt_stress_v,real *lpt_stress_w,real *scg);

//About Swap, and reference, coordiSys is the system direction, valType is the plane direction, incGhost is whether include ghost boundary or not
void block_thread_cell(dim3 &dimBlocks,dim3 &numBlocks,dom_struct dom,int coordiSys,int valType);

//About Swap, and reference, dirc is the system direction, get 3d blocks and threads
void block_thread_cell_3D(dim3 &dimBlocks,dim3 &numBlocks,dom_struct dom,int coordiSys);

//Allocate block and thread for npoints
void block_thread_point(dim3 &dimBlocks,dim3 &numBlocks,int npoints);


/*
Interpolate flow field property at the particle position, it's for cell-center system
	A is cell-centerred flow field data 
	Ag is the property after interpolation, located at each particle
*/
__global__ void interpolate_point_scalar_Lag2(int npoints,real *A,real *Ag, point_struct *points, dom_struct *dom);

__global__ void interpolate_point_scalar_Gaussian(int npoints,real *A,real *Ag, point_struct *points, dom_struct *dom);

__global__ void interpolate_point_scalar_Lag3(int npoints,real *A,real *Ag, point_struct *points, dom_struct *dom);
/*
Interpolate flow field property at the particle position, it's for face-center system
	u,v,w are face-centerred values, can be flow velocity field	
	ug,vg,wg are values after the interpolation, located at the particle position
	bc is the boundary condition for particles, USE PERIODIC BC FOR THE TIME BEING
*/
__global__ void interpolate_point_vel_Lag2(real *u, real *v, real *w,int npoints,real rho_f, real nu,real *ug,real *vg,real *wg, point_struct *points, dom_struct *dom, BC bc);


/*
calculate fluid stress  \nabla \cdot \sigma =-\nabla P +\mu * nabla^2 u
	u,v,w are velocity field at face center
	flag_u~w are tags which indicate interior or boundary of the domain.  flag=Dirichlet（inside particle）, Neuman（domain boundary）, otherwise at interiol of the domain
	dt and dt0 are for Adam-bashforth method in time, not used yet
*/
__global__ void stress_u(real rho_f, real nu, real *u0,real *p,real *p0,real *stress, dom_struct *dom,int *flag_u,real dt,real dt0);

__global__ void stress_v(real rho_f, real nu, real *v0,real *p,real *p0,real *stress, dom_struct *dom,int *flag_v,real dt,real dt0);

__global__ void stress_w(real rho_f, real nu, real *w0,real *p,real *p0,real *stress, dom_struct *dom,int *flag_w,real dt,real dt0);


/*
Calculate fluid force acted on particles, include drag force,fluid stress, added mass effect. Also calculate the exchange rate of soluble scalar mass between particle and flow field.
	ug,vg,wg are the fluid velocity at the particle location
	lpt_stress_u~w are the fluid stresses at the particle location
	gradP is the body force act on fluid
	g is gravity act on particle and fluid.
	C_add,C_stress,C_drag are coefficients of added mass,fluid stress, and drag force. Default values are 0.5,1,1 correspondingly
	sc_eq is the saturation concentration of the scalar at the particle-fluid interface
	DIFF is the diffusivity of the scalar
*/
__global__ void drag_points(point_struct *points, int npoints,
real *ug,real *vg,real *wg,
real *lpt_stress_u,real *lpt_stress_v,real *lpt_stress_w,real *scg,
real rho_f,real mu, g_struct g,gradP_struct gradP,
real C_add,real C_stress,real C_drag,
real sc_eq,real DIFF);

//advance particle position based on Euler prediction correction, it can be divided into 2 steps. See "Predictor–corrector method" in wiki for ref
__global__ void move_points_a(point_struct *points, int npoints, real dt);
__global__ void move_points_b(dom_struct *dom,point_struct *points, int npoints, real dt);


__global__ void drag_move_points(point_struct *points,dom_struct *dom, int npoints,
real *ug,real *vg,real *wg,
real *lpt_stress_u,real *lpt_stress_v,real *lpt_stress_w,real *scg,
real rho_f,real mu, g_struct g,gradP_struct gradP,
real C_add,real C_stress,real C_drag,
real sc_eq,real DIFF,real dt,int *exist,real rinit,real h,real pef);


//Locate the grid cell index (i,j,k) of each particle
__global__ void lpt_localize(int npoints, point_struct *points, dom_struct *dom, BC bc);

__global__ void point_copy(point_struct *points_buf,point_struct *old_points,int *exist,int *shift,int npoints);

__global__ void drag_move_points_init(point_struct *points,dom_struct *dom, int npoints,
real *ug,real *vg,real *wg,
real *lpt_stress_u,real *lpt_stress_v,real *lpt_stress_w,
real rho_f,real mu, g_struct g,gradP_struct gradP,
real C_add,real C_stress,real C_drag,
real sc_eq,real DIFF);




/****f* cuda_particle_kernel/reset_flag_u<<<>>>()
 * NAME
 *  reset_flag_u<<<>>>()
 * USAGE
 */
__global__ void reset_flag_u(int *flag_u, dom_struct *dom, BC bc);
/*
 * FUNCTION
 *  Set all flag_u nodes to fluid (= 1).
 * ARGUMENTS
 *  * flag_u -- the device x-direction velocity flag array subdomain
 *  * dom -- the device subdomain
 *  * bc -- the boundary conditions
 ******
 */

/****f* cuda_particle_kernel/reset_flag_v<<<>>>()
 * NAME
 *  reset_flag_v<<<>>>()
 * USAGE
 */
__global__ void reset_flag_v(int *flag_v, dom_struct *dom, BC bc);
/*
 * FUNCTION
 *  Set all flag_v nodes to fluid (= 1).
 * ARGUMENTS
 *  * flag_v -- the device y-direction velocity flag array subdomain
 *  * dom -- the device subdomain
 *  * bc -- the boundary conditions
 ******
 */

/****f* cuda_particle_kernel/reset_flag_w<<<>>>()
 * NAME
 *  reset_flag_w<<<>>>()
 * USAGE
 */
__global__ void reset_flag_w(int *flag_w, dom_struct *dom, BC bc);
/*
 * FUNCTION
 *  Set all flag_w nodes to fluid (= 1).
 * ARGUMENTS
 *  * flag_w -- the device z-direction velocity flag array subdomain
 *  * dom -- the device subdomain
 *  * bc -- the boundary conditions
 ******
 */

__global__ void generator_array_initD(int npoints,int *A,int *B);

/****f* cuda_particle_kernel/cage_flag_u_periodic_W<<<>>>()
 * NAME
 *  cage_flag_u_periodic_W<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_periodic_W(int *flag_u, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain W boundary flag_u for periodic conditions.
 * ARGUMENTS
 *  * flag_u -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_u_periodic_E<<<>>>()
 * NAME
 *  cage_flag_u_periodic_E<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_periodic_E(int *flag_u, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain E boundary flag_u for periodic conditions.
 * ARGUMENTS
 *  * flag_u -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_u_periodic_S<<<>>>()
 * NAME
 *  cage_flag_u_periodic_S<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_periodic_S(int *flag_u, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain S boundary flag_u for periodic conditions.
 * ARGUMENTS
 *  * flag_u -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_u_periodic_N<<<>>>()
 * NAME
 *  cage_flag_u_periodic_N<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_periodic_N(int *flag_u, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain N boundary flag_u for periodic conditions.
 * ARGUMENTS
 *  * flag_u -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_u_periodic_B<<<>>>()
 * NAME
 *  cage_flag_u_periodic_B<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_periodic_B(int *flag_u, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain B boundary flag_u for periodic conditions.
 * ARGUMENTS
 *  * flag_u -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_u_periodic_T<<<>>>()
 * NAME
 *  cage_flag_u_periodic_T<<<>>>()
 * USAGE
 */
__global__ void cage_flag_u_periodic_T(int *flag_u, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain T boundary flag_u for periodic conditions.
 * ARGUMENTS
 *  * flag_u -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_periodic_W<<<>>>()
 * NAME
 *  cage_flag_v_periodic_W<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_periodic_W(int *flag_v, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain W boundary flag_v for periodic conditions.
 * ARGUMENTS
 *  * flag_v -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_periodic_E<<<>>>()
 * NAME
 *  cage_flag_v_periodic_E<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_periodic_E(int *flag_v, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain E boundary flag_v for periodic conditions.
 * ARGUMENTS
 *  * flag_v -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_periodic_S<<<>>>()
 * NAME
 *  cage_flag_v_periodic_S<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_periodic_S(int *flag_v, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain S boundary flag_v for periodic conditions.
 * ARGUMENTS
 *  * flag_v -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_periodic_N<<<>>>()
 * NAME
 *  cage_flag_v_periodic_N<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_periodic_N(int *flag_v, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain N boundary flag_v for periodic conditions.
 * ARGUMENTS
 *  * flag_v -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_periodic_B<<<>>>()
 * NAME
 *  cage_flag_v_periodic_B<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_periodic_B(int *flag_v, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain B boundary flag_v for periodic conditions.
 * ARGUMENTS
 *  * flag_v -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_v_periodic_T<<<>>>()
 * NAME
 *  cage_flag_v_periodic_T<<<>>>()
 * USAGE
 */
__global__ void cage_flag_v_periodic_T(int *flag_v, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain T boundary flag_v for periodic conditions.
 * ARGUMENTS
 *  * flag_v -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_periodic_W<<<>>>()
 * NAME
 *  cage_flag_w_periodic_W<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_periodic_W(int *flag_w, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain W boundary flag_w for periodic conditions.
 * ARGUMENTS
 *  * flag_w -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_periodic_E<<<>>>()
 * NAME
 *  cage_flag_w_periodic_E<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_periodic_E(int *flag_w, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain E boundary flag_w for periodic conditions.
 * ARGUMENTS
 *  * flag_w -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_periodic_S<<<>>>()
 * NAME
 *  cage_flag_w_periodic_S<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_periodic_S(int *flag_w, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain S boundary flag_w for periodic conditions.
 * ARGUMENTS
 *  * flag_w -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_periodic_N<<<>>>()
 * NAME
 *  cage_flag_w_periodic_N<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_periodic_N(int *flag_w, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain N boundary flag_w for periodic conditions.
 * ARGUMENTS
 *  * flag_w -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_periodic_B<<<>>>()
 * NAME
 *  cage_flag_w_periodic_B<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_periodic_B(int *flag_w, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain B boundary flag_w for periodic conditions.
 * ARGUMENTS
 *  * flag_w -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */

/****f* cuda_particle_kernel/cage_flag_w_periodic_T<<<>>>()
 * NAME
 *  cage_flag_w_periodic_T<<<>>>()
 * USAGE
 */
__global__ void cage_flag_w_periodic_T(int *flag_w, dom_struct *dom);
/*
 * FUNCTION
 *  Update domain T boundary flag_w for periodic conditions.
 * ARGUMENTS
 *  * flag_w -- the device flag array subdomain
 *  * dom -- the device subdomain
 ******
 */














#endif
