#include "point.h"
#include "scalar.h"
#include<stdlib.h>
#include<string.h>
#include<math.h>
#define f(x, a, e) 1. / 4. * (1 + tanh((a + x) / e)) * (1 + tanh((a - x) / e))

int *flag_u;
int **_flag_u;
int *flag_v;
int **_flag_v;
int *flag_w;
int **_flag_w;

int pps;
int ninit;
int nmax;
int npoints;
int nold;
real ratio;
real rho_0;
real rinit;
real pef;
real gene_length;

real I[1000];
long ID = 0;
int nout;
point_struct *points;
point_struct **_points;
point_struct *points_buf;

void points_read_input(void)
{
  int i;  // iterator
  int fret = 0;
  fret = fret; // prevent compiler warning

  // open configuration file for reading
  char fname[FILE_NAME_SIZE];
  sprintf(fname, "%s/input/point.config", ROOT_DIR);
  FILE *infile = fopen(fname, "r");
  if(infile == NULL) {
    fprintf(stderr, "Could not open file %s\n", fname);
    exit(EXIT_FAILURE);
  }

  // read point_point_particle list
  fret = fscanf(infile,"Gene Speed %d\n", &pps);
  fret = fscanf(infile,"ninit %d\n",&ninit);
  fret = fscanf(infile,"Gene Length %lf\n",&gene_length);
  fret = fscanf(infile,"Parcel Size %lf\n",&ratio);
  fret = fscanf(infile,"Pressure %lf\n",&pef);
  fret = fscanf(infile,"Radius %lf\n",&rinit);
  fret = fscanf(infile,"Density %lf\n",&rho_0);
  npoints=ninit;

  if(npoints<=0) return;

  // allocate point_point_particle list
  points = (point_struct*) malloc(npoints * sizeof(point_struct));
  cpumem += npoints * sizeof(point_struct);

  // read npoints point_point_particles
  for(i = 0; i < npoints; i++) {
    fret = fscanf(infile, "\n");
#ifdef DOUBLE
    fret = fscanf(infile, "r %lf\n", &points[i].r);
    fret = fscanf(infile, "(x, y, z) %lf %lf %lf\n",
      &points[i].x, &points[i].y, &points[i].z);
    fret = fscanf(infile, "rho %lf\n", &rho_0);
    points[i].rho=rho_0*(1+(Dom.ze-points[i].z)*pef);
#else // single precision
    fret = fscanf(infile, "r %f\n", &points[i].r);
    fret = fscanf(infile, "(x, y, z) %f %f %f\n",
      &points[i].x, &points[i].y, &points[i].z);
    fret = fscanf(infile, "rho %f\n", rho_0);
    points[i].rho=rho_0*(1+(Dom.ze-points[i].z)*pef);
#endif
//    fret = fscanf(infile, "rotating %d\n", &points[i].rotating);
  }
  rinit=points[0].r;
  fclose(infile);
}

void points_show_config(void)
{
  int i;  // iterator

  printf("point_particles:\n");
  for(i = 0; i < npoints; i++) {
    printf("  point_particle %d:\n", i);
    printf("    r = %e\n", points[i].r);
    printf("    (x, y, z) = (%e, %e, %e)\n",
      points[i].x, points[i].y, points[i].z);
    printf("    (u, v, w) = (%e, %e, %e)\n",
      points[i].u, points[i].v, points[i].w);
    printf("    (udot, vdot, wdot) = (%e, %e, %e)\n",
      points[i].udot, points[i].vdot, points[i].wdot);
        printf("    (Fx, Fy, Fz) = (%e %e %e)\n",
      points[i].Fx, points[i].Fy, points[i].Fz);
    printf("    rho = %f\n", points[i].rho);
    printf("    ID = %ld\n", points[i].id);
	
  }
  /*
  for(i = 0; i < Dom.xn; i++)
  {
		  printf("I[%d] %f ", i, I[i]);
  }
  */
}


//This subroutine delete the old particle&scalar, and inject new particle and scalar into the flow field based on point.config&&scalar.config
int points_scalar_inject(void)
{

//free points on device and host
      cuda_point_free();
      points_clean();
//free scalar on device and host
      cuda_scalar_free();
      scalar_clean();

//read and initialize points	
      points_read_input();
      int points_init_flag = points_init();
      fflush(stdout);
      if(points_init_flag == EXIT_FAILURE) {
        printf("\nThe initial point_particle configuration is not allowed.\n");
        return EXIT_FAILURE;
      }
//read and initialize scalar
      scalar_read_input();
    // initialize the scalar 
      int scalar_init_flag = scalar_init();
      fflush(stdout);
      if(scalar_init_flag == EXIT_FAILURE) {
        printf("\nThe initial scalar configuration is not allowed.\n");
        return EXIT_FAILURE;
      }

//malloc device memory of scalar and point, and push host data to device
      cuda_scalar_malloc();
      cuda_scalar_push();
      
      cuda_point_malloc();
      cuda_point_push();

if(npoints>0) cuda_flow_stress();

//The domain velocity has already been pushed to device
//Match device point velocity with flow field based on point position, which is copied from host
      match_point_vel_with_flow();
//pull the new point infomation to host
          cuda_point_pull();
//Initialize time again
	ttime=0.f;
//write initial field 
          cuda_dom_pull();
          if(rec_flow_field_dt > 0) {
            cgns_grid();
            cgns_flow_field(rec_flow_field_dt);
            rec_flow_field_stepnum_out++;
//printf("\nrec_flow %d\n", rec_flow_field_stepnum_out);
          }
          if(rec_point_particle_dt > 0) {
            cgns_point_particles(rec_point_particle_dt);
            rec_point_particle_stepnum_out++;
          }
      
         if(rec_scalar_field_dt > 0) {
            cgns_scalar_field(rec_scalar_field_dt);
            rec_scalar_stepnum_out++;
          }
	return EXIT_SUCCESS;        

}


int points_init(void)
{
  int i;  // iterators

  for(i = 0; i < npoints; i++) {



    // initialize velocity and acceleration to zero
    points[i].u = 0.;
    points[i].v = 0.;
    points[i].w = 0.;

    points[i].u0 = 0.;
    points[i].v0 = 0.;
    points[i].w0 = 0.;

    points[i].udot = 0.;
    points[i].vdot = 0.;
    points[i].wdot = 0.;
    /* set initial position of point_point_particle reference basis to match the global
     * domain basis */

    // initialize the hydrodynamic forces and moments to zero
    points[i].Fx = 0.;
    points[i].Fy = 0.;
    points[i].Fz = 0.;
    // initialize the point_point_particle interaction force to zero
    points[i].iFx = 0.;
    points[i].iFy = 0.;
    points[i].iFz = 0.;
//TODO  change ms initialization in the future
 //   points[i].ms = 4*PI/3.0f *points[i].r*points[i].r*points[i].r*points[i].rho;

    real m=4*PI/3.0f *points[i].r*points[i].r*points[i].r*points[i].rho;
    points[i].ms = m;

	//printf("\npoint %d %f %f %f %f\n",i,sc_init_percent,m,points[i].r,points[i].rho);

  //  points[i].x=(float)rand()/(float)(RAND_MAX/2)-1;
  //  points[i].y=(float)rand()/(float)(RAND_MAX/2)-1;
  //  points[i].z=0;
    points[i].x0 =points[i].x;
    points[i].y0 =points[i].y;
    points[i].z0 =points[i].z;
    points[i].ms0 =points[i].ms ;
    points[i].msdot =0;
    points[i].hp =0;

    points[i].xi=points[i].x;
    points[i].yi=points[i].y;
    points[i].zi=points[i].z;
//  printf("x y z r rho %f %f %f %f %f\n",points[i].xi,points[i].yi,points[i].zi,points[i].r,points[i].rho);
    //point id
    points[i].id = i+1;
 
   //index of grid that the particle reside in
    points[i].i = 0;
    points[i].j = 0;
    points[i].k = 0;
 
    //point iteration sub-timestep
    points[i].dt = points[i].rho *2.f*points[i].r*points[i].r/(9.0f*mu);
  }
  	sample_init();
	
  return EXIT_SUCCESS;
}

void sample_init(void)
{
		int count, cnum;
		real a, L, dx, e, x;
		L = 1. / 2. * Dom.xl;
		dx = Dom.dx;
		cnum = Dom.xn;
		a = gene_length * L;
		e = a / 5;
		printf("L a e %f %f %f \n", L, gene_length, e);
		for(count = 1; count < cnum; count++)
			{
				x = -L + dx * (count - 1. / 2.);
				I[count] = I[count-1] + f(x, a, e) * dx;
		//		printf("%f \n",f(x,a,e));
			}
		for(count = 0; count < cnum; count++)
			{
				x = -L + dx * (count + 1. / 2.);
				I[count] = I[count] / I[cnum-1] * 2 * L - L;
		//		printf("%f\n",I[count]);
			}
}

real sample(real x)
{
		real L = 1./ 2. * Dom.xl;
		real dx = Dom.dx;
		int i;
		for(i = 0; i < Dom.xn && x > I[i]; i++)
				;
		float x1 = -L + dx * (i - 1. / 2. - 1);
		float x2 = x1 + dx;
		x = (x - I[i-1]) / (I[i] - I[i-1]) * x2 + (I[i] - x) / (I[i] - I[i-1]) * x1;
		return x;
}

void bubble_generate(void)
{
	cuda_point_out();
	int nin = ceil(pps * dt);		
	int nold = npoints;
	npoints += nin - nout;
	printf("nold npoints nin nout %d %d %d %d\n", nold, npoints, nin, nout);
	fflush(stdout);

	point_struct *pointsbuf;
	pointsbuf = (point_struct *) malloc(sizeof(point_struct) * nold);

	memcpy(pointsbuf, points, sizeof(point_struct) * nold);
	printf("Host point to buf\n");
	fflush(stdout);
	free(points);

	points = (point_struct *) malloc(sizeof(point_struct) * npoints);
	memcpy(points, pointsbuf, sizeof(point_struct) * (nold - nout));
	printf("Host buf to point\n");
	fflush(stdout);
	free(pointsbuf);

	//Initialize the new bubbles generated
	for(int i = nold - nout; i < npoints; i++)
	{
		
			points[i].r = rinit;
    		points[i].rho=rho_0*(1+(Dom.ze-points[i].z)*pef);

			points[i].z = 0.;
			points[i].x = (real)rand()/(real)(RAND_MAX/2/Dom.xe) - Dom.xe;
			points[i].x = sample(points[i].x);
			points[i].y = (real)rand()/(real)(RAND_MAX/2/Dom.xe) - Dom.xe;
			points[i].y = sample(points[i].y);

			points[i].u = 0.;
			points[i].v = 0.;
			points[i].w = 0.;

			points[i].u0 = 0.;
			points[i].v0 = 0.;
			points[i].w0 = 0.;

			points[i].udot = 0.;
			points[i].vdot = 0.;
			points[i].wdot = 0.;

			points[i].Fx = 0.;
			points[i].Fy = 0.;
			points[i].Fz = 0.;

			points[i].iFx = 0.;
			points[i].iFy = 0.;
			points[i].iFz = 0.;

			real m=4*PI/3.0f *points[i].r*points[i].r*points[i].r*points[i].rho;
			points[i].ms = m;

			points[i].x0 =points[i].x;
			points[i].y0 =points[i].y;
			points[i].z0 =points[i].z;
			points[i].ms0 =points[i].ms ;
			points[i].msdot =0;
			points[i].hp =0;

			points[i].xi=points[i].x;
			points[i].yi=points[i].y;
			points[i].zi=points[i].z;

			points[i].id = ID;
			ID++;
		 
		   //index of grid that the particle reside in
			points[i].i = 0;
			points[i].j = 0;
			points[i].k = 0;
		 
			//point iteration sub-timestep
			points[i].dt = points[i].rho *2.f*points[i].r*points[i].r/(9.0f*mu);
			
	}
			
	cuda_point_free();
	cuda_point_malloc();
	cuda_point_push();
}

void points_clean(void)
{

  free(points);
}

