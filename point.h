/****h* Bluebottle/point_particle
 * NAME
 *  point_particle
 * FUNCTION
 *  Bluebottle point_particle functions.
 ******
 */

#ifndef _POINT_H
#define _POINT_H

#include "bluebottle.h"

//typedef double float int ireal;

//Determins whether particle will have momentum reaction to fluid flow
//if lpt_twoway >0, there is reaction back to flow, otherwise the particles are passive.
extern real lpt_twoway;

#define EPSILON 1e-7



/****d* bluebottle/SHEAR
 * NAME
 *  SHEAR
 * TYPE
 */
#define POISEUILLE 2
#define OSCILLATORY 3
#define BULK 4
#define TURB 5
#define TAYLOR 6
#define DIFFUSION 7
#define EVEN 8

/*
 * PURPOSE
 *  Define the initial condition shear case.
 ******
 */


//fluid stress located at face center
//stress_u=du/dt =mu * Laplacian(u) -gradP.x
extern real **_stress_u;
extern real **_stress_v;
extern real **_stress_w;

//Fluid vorticity,located on edge center, memory on device
extern real **_omega_x;
extern real **_omega_y;
extern real **_omega_z;


extern float *GaussianKernel; //Gaussian kernel weight contributed by each point particle 
extern int *_DomInfo; //Gaussian kernel weight contributed by each point particle 

//Temp array for particle integration
extern real **ug,**vg,**wg;//device pointer of the fluid velocity at the particle position
extern real **posX,**posY,**posZ;//device pointer of the particle position
extern real **posXold,**posYold,**posZold;//device pointer of the particle position
extern real **lptSourceVal; //Source from each point particle 
extern real **lptSourceValOld; //Source from each point particle 

extern real **lpt_stress_u,**lpt_stress_v,**lpt_stress_w;//device pointer of the fluid velocity at the particle position
extern real **scg;//device pointer of the fluid scalar at the particle position
extern real **Weight; //Gaussian kernel weight contributed by each point particle 
extern real **Ksi; //Gaussian kernel weight contributed by each point particle 
extern int  **cellStart;
extern int  **cellEnd;
extern int  **gridParticleIndex;
extern int  **gridParticleHash;
extern int  **gridFlowHash;
extern int  **pointNumInCell;
extern int  **existStatus;
extern int  **totalout;


/*
Coeff for added mass,fluid stress, drag force, 
the default values are 0.5, 1,1 correspondlingly
*/
extern real C_add;
extern real C_stress;
extern real C_drag;

//Add particle reaction force to fluid momentum source term
void lpt_point_twoway_forcing();

//calculate fluid stress du/dt to impose on particles
void cuda_flow_stress(void);

real cuda_find_dt_points(real dt);

void cuda_find_DIFF_dt_points(void);

/****d* point_particle/NNODES
 * NAME
 *  NNODES
 * TYPE
 */
#define NNODES 26
/*
 * PURPOSE
 *  Define the number of nodes used for the Lebedev quadrature scheme.
 ******
 */

//The Volume fraction of particle and fluid correspondingly

/****v* point_particle/
 * NAME
 *  flag_u
 * TYPE
 */
extern int *flag_u;
/* 
 * PURPOSE
 *  Flag x-direction components of velocity field that are set as boundaries.
 ******
 */

/****v* point_particle/_flag_u
 * NAME
 *  _flag_u
 * TYPE
 */
extern int **_flag_u;
/*
 * PURPOSE
 *  CUDA device analog for flag_u.  It contains pointers to arrays containing
 *  the subdomain fields on which each device operates.
 ******
 */

/****v* point_particle/flag_v
 * NAME
 *  flag_v
 * TYPE
 */
extern int *flag_v;
/* 
 * PURPOSE
 *  Flag y-direction components of velocity field that are set as boundaries.
 ******
 */

/****v* point_particle/_flag_v
 * NAME
 *  _flag_v
 * TYPE
 */
extern int **_flag_v;
/*
 * PURPOSE
 *  CUDA device analog for flag_v.  It contains pointers to arrays containing
 *  the subdomain fields on which each device operates.
 ******
 */

/****v* point_particle/flag_w
 * NAME
 *  flag_w
 * TYPE
 */
extern int *flag_w;
/* 
 * PURPOSE
 *  Flag z-direction components of velocity field that are set as boundaries.
 ******
 */

/****v* point_particle/_flag_w
 * NAME
 *  _flag_w
 * TYPE
 */
extern int **_flag_w;
/*
 * PURPOSE
 *  CUDA device analog for flag_w.  It contains pointers to arrays containing
 *  the subdomain fields on which each device operates.
 ******
 */


/****s* point_particle/point_struct
 * NAME
 *  point_struct
 * TYPE
 */
typedef struct point_struct {
  real r;
  real x;
  real y;
  real z;
  real u;
  real v;
  real w;
  real x0;
  real y0;
  real z0;
  real u0;
  real v0;
  real w0;
  real udot;
  real vdot;
  real wdot;
//initial position for each particle
  real xi;
  real yi;
  real zi;
  /*
  real ox;
  real oy;
  real oz;
  real ox0;
  real oy0;
  real oz0;
  real oxdot;
  real oydot;
  real ozdot;
  */
  real Fx;
  real Fy;
  real Fz;
/*
  real Lx;
  real Ly;
  real Lz;
  real aFx;
  real aFy;
  real aFz;
  real aLx;
  real aLy;
  real aLz;
  real kFx;
  real kFy;
  real kFz;
  */
  real iFx;
  real iFy;
  real iFz;
  /*
  real iLx;
  real iLy;
  real iLz;
  */
 
// density of in-soluble mass in particle or droplet
  real rho;

 //Soluble mass 
  real ms;
  real ms0;

 /*
 Change rate of soluble mass 
 hp*dp/D=2+0.6Re_p^0.5 Sc^{1/3}
 msdot= hp*4*PI*r^2*(sc-sc_sat)
*/
  real msdot;
  real hp;
  real rs;
/*
  real spring_k;
  real spring_x;
  real spring_y;
  real spring_z;
*/
//  int rotating;
  long int id;

//Grid index to record which grid cell the particle center is in
  int i;
  int j;
  int k;

//Sub-timestep for particle velocity integration, not used yet
  real dt;

} point_struct;
/*
 * PURPOSE
 *  Carry physical information regarding a point_particle.
 * MEMBERS
 *  * r -- the point_particle radius size
 *  * x -- the point_particle location component
 *  * y -- the point_particle location component
 *  * z -- the point_particle location component
 *  * u -- linear velocity in x-direction
 *  * v -- linear velocity in y-direction
 *  * w -- linear velocity in z-direction
 *  * u0 -- initial linear velocity in x-direction
 *  * v0 -- initial linear velocity in y-direction
 *  * w0 -- initial linear velocity in z-direction
 *  * udot -- linear acceleration in x-direction
 *  * vdot -- linear acceleration in y-direction
 *  * wdot -- linear acceleration in z-direction
 *  * axx -- x-component of basis vector initially coincident with x-axis
 *  * axy -- y-component of basis vector initially coincident with x-axis
 *  * axz -- z-component of basis vector initially coincident with x-axis
 *  * ayx -- x-component of basis vector initially coincident with y-axis
 *  * ayy -- y-component of basis vector initially coincident with y-axis
 *  * ayz -- z-component of basis vector initially coincident with y-axis
 *  * azx -- x-component of basis vector initially coincident with z-axis
 *  * azy -- y-component of basis vector initially coincident with z-axis
 *  * azz -- z-component of basis vector initially coincident with z-axis
 *  * ox -- angular velocity in x-direction
 *  * oy -- angular velocity in y-direction
 *  * oz -- angular velocity in z-direction
 *  * oxdot -- angular acceleration in x-direction
 *  * oydot -- angular acceleration in y-direction
 *  * ozdot -- angular acceleration in z-direction
 *  * Fx -- hydrodynamic force in the x-direction
 *  * Fy -- hydrodynamic force in the y-direction
 *  * Fz -- hydrodynamic force in the z-direction
 *  * Lx -- hydrodynamic moment in the x-direction
 *  * Ly -- hydrodynamic moment in the y-direction
 *  * Lz -- hydrodynamic moment in the z-direction
 *  * aFx -- applied force in the x-direction
 *  * aFy -- applied force in the y-direction
 *  * aFz -- applied force in the z-direction
 *  * aLx -- applied moment in the x-direction
 *  * aLy -- applied moment in the y-direction
 *  * aLz -- applied moment in the z-direction
 *  * kFx -- applied spring force in the x-direction
 *  * kFy -- applied spring force in the y-direction
 *  * kFz -- applied spring force in the z-direction
 *  * iFx -- interaction force in the x-direction
 *  * iFy -- interaction force in the y-direction
 *  * iFz -- interaction force in the z-direction
 *  * iLx -- interaction moment in the x-direction
 *  * iLy -- interaction moment in the y-direction
 *  * iLz -- interaction moment in the z-direction
 *  * nodes -- the status of the nodes
 *  * rho -- point_particle density
 *  * cage -- the cage_struct that defines the point_particle within the domain
 *  * order -- the order above which to truncate the Lamb's series solution
 *  * rs -- the radius of integration for scalar products
 *  * ncoeff -- the number of Lamb's coefficients required order truncation
 *  * spring_k -- strength of spring pulling point_particle back to origin
 *  * spring_x -- x location of spring connection
 *  * spring_y -- y location of spring connection
 *  * spring_z -- z location of spring connection
 *  * translating -- 1 if allowed to translate, 0 if not
 *  * rotating -- 1 if allowed to rotate, 0 if not
 ******
 */

/****v* point_particle/npoints
 * NAME
 *  npoints
 * TYPE
 */
extern int npoints;
/*
 * PURPOSE
 *  Define the total number of point_particles in the domain.
 ******
 */
extern int pps;//points particles added per second
extern int nmax;//maxium point particles that can be generated into the domain;
extern int ninit;//initial point particles in the domain
extern int nold;
extern real rinit;
//extern real zinit[3];
extern int nout;
extern real rho_0;
extern real ratio;
extern real pef;

/****v* point_particle/points
 * NAME
 *  points
 * TYPE
 */
extern point_struct *points;
/*
 * PURPOSE
 *  A list of all point_particles.
 ******
 */



/****v* point_particle/_points
 * NAME
 *  _points
 * TYPE
 */
extern point_struct **_points;
/*
 * PURPOSE
 *  CUDA device analog for points.  It contains pointers to arrays containing
 *  the point_particles in domain on which each device operates.  NOTE: for now,
 *  point_particles are designed to function on only one device.
 ******
 */
extern point_struct *points_buf;

/****v* bluebottle/rec_point_particle_stepnum_out
 * NAME
 *  rec_point_particle_stepnum_out
 * TYPE
 */
extern int rec_point_particle_stepnum_out;
/*
 * PURPOSE
 *  The output timestep number for the simulation.  The initial configuration
 *  is given by stepnum = 0.
 ******
 */



/****v* bluebottle/rec_point_particle_dt
 * NAME
 *  rec_point_particle_dt
 * TYPE
 */
extern real rec_point_particle_dt;
/*
 * PURPOSE
 *  Recorder point_particle output timestep size.
 ******
 */

/****v* bluebottle/rec_point_particle_pos
 * NAME
 *  rec_point_particle_pos
 * TYPE
 */
extern int rec_point_particle_pos;
/*
 * PURPOSE
 *  Recorder point_particle output position.
 ******
 */

/****v* bluebottle/rec_point_particle_a
 * NAME
 *  rec_point_particle_a
 * TYPE
 */
extern int rec_point_particle_a;
/*
 * PURPOSE
 *  Recorder point_particle output radius.
 ******
 */

/****v* bluebottle/rec_point_particle_vel
 * NAME
 *  rec_point_particle_vel
 * TYPE
 */
extern int rec_point_particle_vel;
/*
 * PURPOSE
 *  Recorder point_particle output velocity.
 ******
 */

/****v* bluebottle/rec_point_particle_omega
 * NAME
 *  rec_point_particle_omega
 * TYPE
 */
extern int rec_point_particle_omega;
/*
 * PURPOSE
 *  Recorder point_particle output angular velocity.
 ******
 */

/****v* bluebottle/rec_point_particle_force
 * NAME
 *  rec_point_particle_force
 * TYPE
 */
extern int rec_point_particle_force;
/*
 * PURPOSE
 *  Recorder point_particle output hydrodynamic force.
 ******
 */

/****v* bluebottle/rec_point_particle_moment
 * NAME
 *  rec_point_particle_moment
 * TYPE
 */
extern int rec_point_particle_moment;
/*
 * PURPOSE
 *  Recorder point_particle output hydrodynamic moment.
 ******
 */


/****f* point_particle/points_read_input()
 * NAME
 *  points_read_input()
 * USAGE
 */
void points_read_input(void);
/*
 * FUNCTION
 *  Read point_particle specfications from point.input.
 * ARGUMENTS
 *  * turb -- boolean operator for removing point_particles from turbulence precursor
 ******
 */

/****f* point_particle/points_show_config()
 * NAME
 *  points_show_config()
 * USAGE
 */
void points_show_config(void);
/*
 * FUNCTION
 *  Write point_particle specifications to screen.
 ******
 */


void points_scalar_inject(void);
void match_point_vel_with_flow(void);

/****f* point_particle/points_init()
 * NAME
 *  points_init()
 * USAGE
 *
 */
int points_init(void);
/*
 * FUNCTION
 *  Initialize the point_particles (build initial cages) and phase.
 * RESULT
 *  EXIT_SUCCESS if successful, EXIT_FAILURE otherwise.
 ******
 */
void bubble_generate(void);

void sample_init(void);

real sample(real x);


/****f* point_particle/flags_reset()
 * NAME
 *  flags_reset()
 * USAGE
 */
void flags_reset(void);
/*
 * FUNCTION
 *  Reinitializes flag arrays to no boundaries (1).
 ******
 */



/****f* bluebottle/cuda_point_malloc()
 * NAME
 *  cuda_point_malloc()
 * USAGE
 */
void cuda_point_malloc(void);
/*
 * FUNCTION
 *  Allocate device memory reference pointers on host and device memory on
 *  device for the point_particles.
 ******
 */


/****f* bluebottle/cuda_point_push()
 * NAME
 *  cuda_point_push()
 * USAGE
 */
void cuda_point_push(void);
/*
 * FUNCTION
 *  Copy point_particle data from host to device.
 ******
 */




/****f* bluebottle/cuda_point_pull()
 * NAME
 *  cuda_point_pull()
 * USAGE
 */
void cuda_point_pull(void);
/*
 * FUNCTION
 *  Copy point_particle data from device to host.
 ******
 */

/****f* bluebottle/cuda_point_free()
 * NAME
 *  cuda_point_free()
 * USAGE
 */
void cuda_point_free(void);
/*
 * FUNCTION
 *  Free device memory for the point_particles on device and device memory reference
 *  pointers on host.
 ******
 */


//return volume fraction of point particles
//real cuda_points_volFrac();


/****f* point_particle/points_clean()
 * NAME
 *  points_clean()
 * USAGE
 */
void points_clean(void);
/*
 * FUNCTION
 *  Clean up.  Free any allocated host memory.
 ******
 */



//void cgns_point_particles(real dtout);

/****f* bluebottle/cuda_move_points()
 * NAME
 *  cuda_move_points()
 * USAGE
 */

//move point particles
void cuda_move_points(void);
/*
 * FUNCTION
 *  Calculate new point_particle velocities and positions.
 ******
 */

void cuda_point_out(void);
/*FUNCTION
 *Calculate the bubble generator array and update number of bubbles inside the domain
 *
 */
//void create_point_buf(void);
/*FUNCTION
 * Malloc memory for point_buf and push from host,finally copy information from points
 */


//void cuda_add_points(void);
/*FUNCTION
 * copy from buf to points to finalize the bubble generating process
 */


void generator_array_initH();
/*
typedef struct grid_info {
0  int is;
1  int ie;
2  int in;
3  int isb;
4  int ieb;
5  int inb;
6  int js;
7  int je;
8  int jn;
9  int jsb;
10  int jeb;
11  int jnb;
12  int ks;
13  int ke;
14  int kn;
15  int ksb;
16  int keb;
17  int knb;
18  int s1;
19  int s2;
20  int s3;
21  int s1b;
22  int s2b;
23  int s3b;

4  int _is;
  int _ie;
  int _in;
  int _isb;
  int _ieb;
  int _inb;
5  int _js;
  int _je;
  int _jn;
  int _jsb;
  int _jeb;
  int _jnb;
6  int _ks;
  int _ke;
  int _kn;
  int _ksb;
  int _keb;
  int _knb;
7  int _s1;
  int _s1b;
  int _s2;
  int _s2b;
  int _s3;
  int _s3b;
} grid_info;
*/
#endif
