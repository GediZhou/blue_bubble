#include "cuda_scalar.h"
#include "cuda_point.h"
// scalar; west; periodic
__global__ void diffScalar_explicitD(real *sc,real *sc0,
dom_struct *dom, real DIFF_dt)
{
//store scalar value of difference direction at cell center
__shared__ real sc_b[MAX_THREADS_DIM * MAX_THREADS_DIM]; // sc at bottom center
__shared__ real sc_t[MAX_THREADS_DIM * MAX_THREADS_DIM]; // sc at top center
__shared__ real sc_c[MAX_THREADS_DIM * MAX_THREADS_DIM]; // sc at center
real ddx = 1. / dom->dx; // to limit the number of divisions needed
real ddy = 1. / dom->dy; // to limit the number of divisions needed
real ddz = 1. / dom->dz; // to limit the number of divisions needed
int C;
// loop over z-planes
// for(int k = dom->Gcc._ksb; k < dom->Gcc._keb; k++) 
for(int k = dom->Gcc._ks; k < dom->Gcc._ke; k++) {
// subdomain indices
// the extra 2*blockIdx.X terms implement the necessary overlapping of
// shared memory blocks in the subdomain
int i = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
int j = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;
// shared memory indices
int ti = threadIdx.x;
int tj = threadIdx.y;
// load shared memory
// TODO: look into the effect of removing these if statements and simply
// allowing memory overruns for threads that don't matter for particular
// discretizations
if( j < dom->Gcc._jeb&& i < dom->Gcc._ieb) {
sc_c[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
sc_b[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k-1)*dom->Gcc._s2b];
sc_t[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k+1)*dom->Gcc._s2b];
}
// make sure all threads complete shared memory copy
__syncthreads();
// compute convective term
// if off the shared memory block boundary
if((ti > 0 && ti < blockDim.x-1) && (tj > 0 && tj < blockDim.y-1)) {
//diffusive term of scalar at cell center
real ddsdxx=(sc_c[ti-1 + tj*blockDim.x]+sc_c[ti+1 + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddx* ddx;
real ddsdyy=(sc_c[ti + (tj-1)*blockDim.x]+sc_c[ti + (tj+1)*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddy* ddy;
real ddsdzz=(sc_b[ti + tj*blockDim.x]+sc_t[ti + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddz* ddz;
real sc_diff= (ddsdxx+ddsdyy+ddsdzz)*DIFF_dt;
sc_c[ti + tj*blockDim.x] +=sc_diff;
}
// copy shared memory back to global, without copying boundary ghost values
//sc stores n+1 timestep, conv and diff are n timestep!!
if((j >= dom->Gcc._js && j < dom->Gcc._je)
&& (i >= dom->Gcc._is && i < dom->Gcc._ie)
&& (ti > 0 && ti < (blockDim.x-1))
&& (tj > 0 && tj < (blockDim.y-1))) {
C = i + j*dom->Gcc._s1b + k*dom->Gcc._s2b;
sc[C] = sc_c[ti + tj*blockDim.x];
}
}
}

__global__ void BC_sc_W_P(real *sc, dom_struct *dom)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((tj < dom->Gcc._jnb) && (tk < dom->Gcc._knb))
    sc[dom->Gcc._isb + tj*s1b + tk*s2b] = sc[(dom->Gcc._ie-1) + tj*s1b + tk*s2b];
}

// scalar; west; Neumann
__global__ void BC_sc_W_N(real *sc, dom_struct *dom)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((tj < dom->Gcc._jnb) && (tk < dom->Gcc._knb))
    sc[dom->Gcc._isb + tj*s1b + tk*s2b] = sc[dom->Gcc._is + tj*s1b + tk*s2b];
}

// scalar; east; periodic
__global__ void BC_sc_E_P(real *sc, dom_struct *dom)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((tj < dom->Gcc._jnb) && (tk < dom->Gcc._knb))
    sc[(dom->Gcc._ieb-1) + tj*s1b + tk*s2b] = sc[dom->Gcc._is + tj*s1b + tk*s2b];
}

// scalar; east; Neumann
__global__ void BC_sc_E_N(real *sc, dom_struct *dom)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((tj < dom->Gcc._jnb) && (tk < dom->Gcc._knb))
    sc[(dom->Gcc._ieb-1) + tj*s1b + tk*s2b] = sc[(dom->Gcc._ie-1)
      + tj*s1b + tk*s2b];
}

// scalar; south; periodic
__global__ void BC_sc_S_P(real *sc, dom_struct *dom)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tk < dom->Gcc._knb))
    sc[ti + dom->Gcc._jsb*s1b + tk*s2b] = sc[ti + (dom->Gcc._je-1)*s1b + tk*s2b];
}

// scalar; south; Neumann
__global__ void BC_sc_S_N(real *sc, dom_struct *dom)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tk < dom->Gcc._knb))
    sc[ti + dom->Gcc._jsb*s1b + tk*s2b] = sc[ti + dom->Gcc._js*s1b + tk*s2b];
}

// scalar; north; periodic
__global__ void BC_sc_N_P(real *sc, dom_struct *dom)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tk < dom->Gcc._knb))
    sc[ti + (dom->Gcc._jeb-1)*s1b + tk*s2b] = sc[ti + dom->Gcc._js*s1b + tk*s2b];
}

// scalar; north; Neumann
__global__ void BC_sc_N_N(real *sc, dom_struct *dom)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tk < dom->Gcc._knb))
    sc[ti + (dom->Gcc._jeb-1)*s1b + tk*s2b] = sc[ti
      + (dom->Gcc._je-1)*s1b + tk*s2b];
}

// scalar; bottom; periodic
__global__ void BC_sc_B_P(real *sc, dom_struct *dom)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tj < dom->Gcc._jnb))
    sc[ti + tj*s1b + dom->Gcc._ksb*s2b] = sc[ti + tj*s1b + (dom->Gcc._ke-1)*s2b];
}

// scalar; bottom; Neumann
__global__ void BC_sc_B_N(real *sc, dom_struct *dom)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b; 
  if((ti < dom->Gcc._inb) && (tj < dom->Gcc._jnb))
    sc[ti + tj*s1b + dom->Gcc._ksb*s2b] = sc[ti + tj*s1b + dom->Gcc._ks*s2b];
}

// scalar; top; periodic
__global__ void BC_sc_T_P(real *sc, dom_struct *dom)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tj < dom->Gcc._jnb))
    sc[ti + tj*s1b + (dom->Gcc._keb-1)*s2b] = sc[ti + tj*s1b + dom->Gcc._ks*s2b];
}

// scalar; top; Neumann
__global__ void BC_sc_T_N(real *sc, dom_struct *dom)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tj < dom->Gcc._jnb))
    sc[ti + tj*s1b + (dom->Gcc._keb-1)*s2b] = sc[ti
      + tj*s1b + (dom->Gcc._ke-1)*s2b];
}


// scalar; west; Dirichlet
__global__ void BC_sc_W_D(real *sc, dom_struct *dom, real bc)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((tj < dom->Gcc._jnb) && (tk < dom->Gcc._knb))
    sc[dom->Gcc._isb + tj*s1b + tk*s2b] = 2*bc-sc[dom->Gcc._is + tj*s1b + tk*s2b];
}


// scalar; east; Dirichlet
__global__ void BC_sc_E_D(real *sc, dom_struct *dom, real bc)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((tj < dom->Gcc._jnb) && (tk < dom->Gcc._knb))
    sc[(dom->Gcc._ieb-1) + tj*s1b + tk*s2b] = 2*bc-sc[(dom->Gcc._ie-1) + tj*s1b + tk*s2b];
}

// scalar; south; Dirichlet
__global__ void BC_sc_S_D(real *sc, dom_struct *dom, real bc)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tk < dom->Gcc._knb))
     sc[ti + dom->Gcc._jsb*s1b + tk*s2b] = 2 * bc - sc[ti + dom->Gcc._js*s1b + tk*s2b];
}

// scalar; north; Dirichlet
__global__ void BC_sc_N_D(real *sc, dom_struct *dom, real bc)
{
  int tk = blockDim.x*blockIdx.x + threadIdx.x;
  int ti = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tk < dom->Gcc._knb))
     sc[ti + (dom->Gcc._jeb-1)*s1b + tk*s2b] = 2 * bc - sc[ti + (dom->Gcc._je-1)*s1b + tk*s2b];
}

// scalar; bottom; Dirichlet
__global__ void BC_sc_B_D(real *sc, dom_struct *dom, real bc)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tj < dom->Gcc._jnb))
    sc[ti + tj*s1b + dom->Gcc._ksb*s2b] = 2 * bc - sc[ti + tj*s1b + dom->Gcc._ks*s2b];

}

// scalar; top; Dirichlet
__global__ void BC_sc_T_D(real *sc, dom_struct *dom, real bc)
{
  int ti = blockDim.x*blockIdx.x + threadIdx.x;
  int tj = blockDim.y*blockIdx.y + threadIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;

  if((ti < dom->Gcc._inb) && (tj < dom->Gcc._jnb))
    sc[ti + tj*s1b + (dom->Gcc._keb-1)*s2b] = 2 * bc - sc[ti + tj*s1b + (dom->Gcc._ke-1)*s2b];
}

//QUICK scheme, this one only deal with periodic BC, Doesn't use Adam-Bashforth for the time being
__global__ void advance_sc_QUICK(real DIFF, 
				real *u, real *v, real *w,
				real *f,real *epsp,
				real *diff, real *conv, 
				real *sc,real *sc0,
				dom_struct *dom, real dt)
{
  // create shared memory
  // no reason to load pressure into shared memory, but leaving it in global
  // will require additional if statements, so keep it in shared
  __shared__ real s_w0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w back
  __shared__ real s_w1[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w center
  __shared__ real s_u[MAX_THREADS_DIM * MAX_THREADS_DIM];       // u
  __shared__ real s_v[MAX_THREADS_DIM * MAX_THREADS_DIM];       // v
//store scalar related values such as convective and diffusive term.
  __shared__ real s_d[MAX_THREADS_DIM * MAX_THREADS_DIM];       // diff
  __shared__ real s_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // conv
  __shared__ real s_f[MAX_THREADS_DIM * MAX_THREADS_DIM];       // source term


//store scalar value of difference direction at cell center
  __shared__ real sc_b[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at bottom center
  __shared__ real sc_t[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at top	 center
 __shared__ real sc_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at  	 center

  __shared__ real sc_ee[MAX_THREADS_DIM * MAX_THREADS_DIM];       
  __shared__ real sc_ww[MAX_THREADS_DIM * MAX_THREADS_DIM];       
  __shared__ real sc_ss[MAX_THREADS_DIM * MAX_THREADS_DIM];       
  __shared__ real sc_nn[MAX_THREADS_DIM * MAX_THREADS_DIM];       
  __shared__ real sc_bb[MAX_THREADS_DIM * MAX_THREADS_DIM];       
  __shared__ real sc_tt[MAX_THREADS_DIM * MAX_THREADS_DIM];       
 
//store point volume percentage in each grid cell
  __shared__ real p_epsp[MAX_THREADS_DIM * MAX_THREADS_DIM];       
 


  real ddx = 1. / dom->dx;     // to limit the number of divisions needed
  real ddy = 1. / dom->dy;     // to limit the number of divisions needed
  real ddz = 1. / dom->dz;     // to limit the number of divisions needed

 
  int C;

  // loop over z-planes
//  for(int k = dom->Gcc._ksb; k < dom->Gcc._keb; k++) {
  for(int k = dom->Gcc._ks; k < dom->Gcc._ke; k++) {
    // subdomain indices
    // the extra 2*blockIdx.X terms implement the necessary overlapping of
    // shared memory blocks in the subdomain
    int i = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;

    // shared memory indices
    int ti = threadIdx.x;
    int tj = threadIdx.y;

    // load shared memory
    // TODO: look into the effect of removing these if statements and simply
    // allowing memory overruns for threads that don't matter for particular
    // discretizations

    if((j >= dom->Gcc._jsb && j < dom->Gcc._jeb)&& (i >= dom->Gcc._isb && i < dom->Gcc._ieb)) {
sc_c[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
sc_b[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k-1)*dom->Gcc._s2b];
sc_t[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k+1)*dom->Gcc._s2b];

//Peridic BC
int ww=i-2;
int ee=i+2;
if(ww<0) 		ww=dom->Gcc._ie-2;
if(ee>=dom->Gcc._ie)    ee=dom->Gcc._is+1;

int ss=j-2;
int nn=j+2;
if(ss<0) 		ss=dom->Gcc._je-2;
if(nn>=dom->Gcc._je)    nn=dom->Gcc._js+1;

int bb=k-2;
int tt=k+2;
if(bb<0) 		bb=dom->Gcc._ke-2;
if(tt>=dom->Gcc._ke)    tt=dom->Gcc._ks+1;

sc_ee[ti + tj*blockDim.x]=sc0[ee+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
sc_ww[ti + tj*blockDim.x]=sc0[ww+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
sc_ss[ti + tj*blockDim.x]=sc0[i+ss*dom->Gcc._s1b + k*dom->Gcc._s2b];
sc_nn[ti + tj*blockDim.x]=sc0[i+nn*dom->Gcc._s1b + k*dom->Gcc._s2b];
sc_bb[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + bb*dom->Gcc._s2b];
sc_tt[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + tt*dom->Gcc._s2b];

s_f[ti + tj*blockDim.x]=    f[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
p_epsp[ti + tj*blockDim.x]= epsp[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
}

   // make sure all threads complete shared memory copy
    __syncthreads();

    if((j >= dom->Gfz._js && j < dom->Gfz._je)
      && (i >= dom->Gfz._is && i < dom->Gfz._ie)) {
      s_w0[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + k*dom->Gfz._s2b];
      s_w1[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + (k+1)*dom->Gfz._s2b];
    }
    if((j >= dom->Gfx._js && j < dom->Gfx._je)
      && (i >= dom->Gfx._is && i < dom->Gfx._ie)) {
      s_u[ti + tj*blockDim.x] = u[i + j*dom->Gfx._s1b + k*dom->Gfx._s2b];
    }
    if((j >= dom->Gfy._js && j < dom->Gfy._je)
      && (i >= dom->Gfy._is && i < dom->Gfy._ie)) {
      s_v[ti + tj*blockDim.x] = v[i + j*dom->Gfy._s1b + k*dom->Gfy._s2b];
    }

  
    // make sure all threads complete shared memory copy
    __syncthreads();

    // compute convective term
    // if off the shared memory block boundary
    if((ti > 0 && ti < blockDim.x-1) && (tj > 0 && tj < blockDim.y-1)){
/*
//scalar at face center
real sc_uw=(sc_c[ti + tj*blockDim.x]+sc_c[ti-1 + tj*blockDim.x])/2.0f;
real sc_ue=(sc_c[ti + tj*blockDim.x]+sc_c[ti+1 + tj*blockDim.x])/2.0f;

real sc_vs=(sc_c[ti + tj*blockDim.x]+sc_c[ti + (tj-1)*blockDim.x])/2.0f;
real sc_vn=(sc_c[ti + tj*blockDim.x]+sc_c[ti + (tj+1)*blockDim.x])/2.0f;

real sc_wb=(sc_b[ti + tj*blockDim.x]+sc_c[ti + tj*blockDim.x])/2.0f;
real sc_wt=(sc_t[ti + tj*blockDim.x]+sc_c[ti + tj*blockDim.x])/2.0f;
*/

//diffusive term of scalar at cell center--CenterScheme
real ddsdxx=(sc_c[ti-1 + tj*blockDim.x]+sc_c[ti+1 + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddx* ddx;
real ddsdyy=(sc_c[ti + (tj-1)*blockDim.x]+sc_c[ti + (tj+1)*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddy* ddy;
real ddsdzz=(sc_b[ti + tj*blockDim.x]+sc_t[ti + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddz* ddz;

//Convective term using QUICK scheme,  How to treat the ti+2,ti-2 etc in shared memory????? 
real ue_plus=fmax(s_u[ti+1 + tj*blockDim.x],0);
real ue_minus=fmin(s_u[ti+1 + tj*blockDim.x],0);
real uw_plus=fmax(s_u[ti + tj*blockDim.x],0);
real uw_minus=fmin(s_u[ti + tj*blockDim.x],0);

real vn_plus=fmax(s_v[ti + (tj+1)*blockDim.x],0);
real vn_minus=fmin(s_v[ti + (tj+1)*blockDim.x],0);
real vs_plus=fmax(s_v[ti + tj*blockDim.x],0);
real vs_minus=fmin(s_v[ti + tj*blockDim.x],0);

real wt_plus=fmax(s_w1[ti + tj*blockDim.x],0);
real wt_minus=fmin(s_w1[ti + tj*blockDim.x],0);
real wb_plus=fmax(s_w0[ti + tj*blockDim.x],0);
real wb_minus=fmin(s_w0[ti + tj*blockDim.x],0);

real phi_ep= 0.75f*sc_c[ti + tj*blockDim.x]+0.375f*sc_c[ti+1 + tj*blockDim.x]-0.125f*sc_c[ti-1 + tj*blockDim.x];
real phi_wp= 0.75f*sc_c[ti-1 + tj*blockDim.x]+0.375f*sc_c[ti + tj*blockDim.x]-0.125f*sc_ww[ti + tj*blockDim.x];
real phi_wn= 0.75f*sc_c[ti + tj*blockDim.x]+0.375f*sc_c[ti-1 + tj*blockDim.x]-0.125f*sc_c[ti+1 + tj*blockDim.x];
real phi_en= 0.75f*sc_c[ti+1 + tj*blockDim.x]+0.375f*sc_c[ti + tj*blockDim.x]-0.125f*sc_ee[ti + tj*blockDim.x];

real phi_np= 0.75f*sc_c[ti + tj*blockDim.x]+0.375f*sc_c[ti + (tj+1)*blockDim.x]-0.125f*sc_c[ti + (tj-1)*blockDim.x];
real phi_sp= 0.75f*sc_c[ti + (tj-1)*blockDim.x]+0.375f*sc_c[ti + tj*blockDim.x]-0.125f*sc_ss[ti + tj*blockDim.x];
real phi_sn= 0.75f*sc_c[ti + tj*blockDim.x]+0.375f*sc_c[ti + (tj-1)*blockDim.x]-0.125f*sc_c[ti + (tj+1)*blockDim.x];
real phi_nn= 0.75f*sc_c[ti + (tj+1)*blockDim.x]+0.375f*sc_c[ti + tj*blockDim.x]-0.125f*sc_nn[ti + tj*blockDim.x];

real phi_tp= 0.75f*sc_c[ti + tj*blockDim.x]+0.375f*sc_t[ti + tj*blockDim.x]-0.125f*sc_b[ti + tj*blockDim.x];
real phi_bp= 0.75f*sc_b[ti + tj*blockDim.x]+0.375f*sc_c[ti + tj*blockDim.x]-0.125f*sc_bb[ti + tj*blockDim.x];
real phi_bn= 0.75f*sc_c[ti + tj*blockDim.x]+0.375f*sc_b[ti + tj*blockDim.x]-0.125f*sc_t[ti + tj*blockDim.x];
real phi_tn= 0.75f*sc_t[ti + tj*blockDim.x]+0.375f*sc_c[ti + tj*blockDim.x]-0.125f*sc_tt[ti + tj*blockDim.x];

real dsudx=((ue_plus*phi_ep+ue_minus*phi_en)-(uw_plus*phi_wp+uw_minus*phi_wn))*ddx;
real dsvdy=((vn_plus*phi_np+vn_minus*phi_nn)-(vs_plus*phi_sp+vs_minus*phi_sn))*ddy;
real dswdz=((wt_plus*phi_tp+wt_minus*phi_tn)-(wb_plus*phi_bp+wb_minus*phi_bn))*ddz;

     //move convective term to right hand side 
      s_c[ti + tj*blockDim.x]=-(dsudx+dsvdy+dswdz);
     //Diffusion term
      s_d[ti + tj*blockDim.x]= (ddsdxx+ddsdyy+ddsdzz)*DIFF;


real sc_conv= s_c[ti + tj*blockDim.x];
real sc_diff= s_d[ti + tj*blockDim.x];

//TODO should we have s_f at t+1/2 step as well? 
//advance scalar. Take the particle volume fraction into consideration
//real rhs=(sc_diff+s_f[ti + tj*blockDim.x])/(1-p_epsp[ti + tj*blockDim.x]);
real rhs=sc_diff+s_f[ti + tj*blockDim.x];
//real rhs=s_f[ti + tj*blockDim.x];
sc_c[ti + tj*blockDim.x] +=(sc_conv+rhs)*dt;

//if(i==30&&j==30&&k==30) printf("\nsc_diff %f %f %f %f\n",sc_diff,sc_conv,s_f[ti + tj*blockDim.x],dt);
}

// copy shared memory back to global, without copying boundary ghost values
//sc stores n+1 timestep, conv and diff are n timestep!!
    if((j >= dom->Gcc._js && j < dom->Gcc._je)
     && (i >= dom->Gcc._is && i < dom->Gcc._ie)
      && (ti > 0 && ti < (blockDim.x-1))
      && (tj > 0 && tj < (blockDim.y-1))) {
      C = i + j*dom->Gcc._s1b + k*dom->Gcc._s2b;
      sc[C] = sc_c[ti + tj*blockDim.x];
      conv[C] = s_c[ti + tj*blockDim.x];
      diff[C] = s_d[ti + tj*blockDim.x];
    }
  }
}

//MacCormac scheme 
__global__ void advance_sc_macCormack(	real DIFF, 
				real *u, real *v, real *w,
				real *f,real *epsp,
				real *diff, real *conv, 
				real *sc,real *sc0,
				dom_struct *dom, real dt)
{
  // create shared memory
  // no reason to load pressure into shared memory, but leaving it in global
  // will require additional if statements, so keep it in shared
  __shared__ real s_w0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w back
  __shared__ real s_w1[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w center
  __shared__ real s_u[MAX_THREADS_DIM * MAX_THREADS_DIM];       // u
  __shared__ real s_v[MAX_THREADS_DIM * MAX_THREADS_DIM];       // v
//store scalar related values such as convective and diffusive term.
  __shared__ real s_d[MAX_THREADS_DIM * MAX_THREADS_DIM];       // diff
  __shared__ real s_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // conv
  __shared__ real s_f[MAX_THREADS_DIM * MAX_THREADS_DIM];       // source term


//store scalar value of difference direction at cell center
  __shared__ real sc_b[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at bottom center
  __shared__ real sc_t[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at top	 center
  __shared__ real sc_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at  	 center

//store point volume percentage in each grid cell
  __shared__ real p_epsp[MAX_THREADS_DIM * MAX_THREADS_DIM];       
 


  real ddx = 1. / dom->dx;     // to limit the number of divisions needed
  real ddy = 1. / dom->dy;     // to limit the number of divisions needed
  real ddz = 1. / dom->dz;     // to limit the number of divisions needed

 
  int C;

  // loop over z-planes
//  for(int k = dom->Gcc._ksb; k < dom->Gcc._keb; k++) {
  for(int k = dom->Gcc._ks; k < dom->Gcc._ke; k++) {
    // subdomain indices
    // the extra 2*blockIdx.X terms implement the necessary overlapping of
    // shared memory blocks in the subdomain
    int i = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;

    // shared memory indices
    int ti = threadIdx.x;
    int tj = threadIdx.y;

    // load shared memory
    // TODO: look into the effect of removing these if statements and simply
    // allowing memory overruns for threads that don't matter for particular
    // discretizations

    if((j >= dom->Gcc._jsb && j < dom->Gcc._jeb)&& (i >= dom->Gcc._isb && i < dom->Gcc._ieb)) {
sc_c[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
sc_b[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k-1)*dom->Gcc._s2b];
sc_t[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k+1)*dom->Gcc._s2b];

s_f[ti + tj*blockDim.x]=    f[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
p_epsp[ti + tj*blockDim.x]= epsp[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
}

   // make sure all threads complete shared memory copy
    __syncthreads();

    if((j >= dom->Gfz._js && j < dom->Gfz._je)
      && (i >= dom->Gfz._is && i < dom->Gfz._ie)) {
      s_w0[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + k*dom->Gfz._s2b];
      s_w1[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + (k+1)*dom->Gfz._s2b];
    }
    if((j >= dom->Gfx._js && j < dom->Gfx._je)
      && (i >= dom->Gfx._is && i < dom->Gfx._ie)) {
      s_u[ti + tj*blockDim.x] = u[i + j*dom->Gfx._s1b + k*dom->Gfx._s2b];
    }
    if((j >= dom->Gfy._js && j < dom->Gfy._je)
      && (i >= dom->Gfy._is && i < dom->Gfy._ie)) {
      s_v[ti + tj*blockDim.x] = v[i + j*dom->Gfy._s1b + k*dom->Gfy._s2b];
    }

  
    // make sure all threads complete shared memory copy
    __syncthreads();

    // compute convective term
    // if off the shared memory block boundary
    if((ti > 0 && ti < blockDim.x-1) && (tj > 0 && tj < blockDim.y-1)){
//scalar at face center
real sc_uw=(sc_c[ti + tj*blockDim.x]+sc_c[ti-1 + tj*blockDim.x])/2.0f;
real sc_ue=(sc_c[ti + tj*blockDim.x]+sc_c[ti+1 + tj*blockDim.x])/2.0f;

real sc_vs=(sc_c[ti + tj*blockDim.x]+sc_c[ti + (tj-1)*blockDim.x])/2.0f;
real sc_vn=(sc_c[ti + tj*blockDim.x]+sc_c[ti + (tj+1)*blockDim.x])/2.0f;

real sc_wb=(sc_b[ti + tj*blockDim.x]+sc_c[ti + tj*blockDim.x])/2.0f;
real sc_wt=(sc_t[ti + tj*blockDim.x]+sc_c[ti + tj*blockDim.x])/2.0f;

//convective term of scalar at cell center
real dsudx=(s_u[(ti+1) + tj*blockDim.x]*sc_ue - s_u[ti + tj*blockDim.x]*sc_uw) * ddx;
real dsvdy=(s_v[ti + (tj+1)*blockDim.x]*sc_vn - s_v[ti + tj*blockDim.x]*sc_vs) * ddy;
real dswdz=(s_w1[ti + tj*blockDim.x]   *sc_wt - s_w0[ti + tj*blockDim.x]*sc_wb) * ddz;


//velocity  at cell center
real u_c=(s_u[(ti+1) + tj*blockDim.x] + s_u[ti + tj*blockDim.x])/2.0f;
real v_c=(s_v[ti + (tj+1)*blockDim.x]+s_v[ti + tj*blockDim.x])/2.0f;
real w_c=(s_w1[ti + tj*blockDim.x]+s_w0[ti + tj*blockDim.x])/2.0f;


//diffusive term of scalar at cell center
real ddsdxx=(sc_c[ti-1 + tj*blockDim.x]+sc_c[ti+1 + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddx* ddx;
real ddsdyy=(sc_c[ti + (tj-1)*blockDim.x]+sc_c[ti + (tj+1)*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddy* ddy;
real ddsdzz=(sc_b[ti + tj*blockDim.x]+sc_t[ti + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddz* ddz;

//Artifical diffusivity for MacCormack scheme,  Using the value on x(i),xm(i),x(i+1)
real ddsdxx_num=(sc_uw*s_u[ti + tj*blockDim.x]*s_u[ti + tj*blockDim.x]+sc_ue*s_u[ti+1 + tj*blockDim.x]*s_u[ti+1 + tj*blockDim.x]-2*u_c*u_c*sc_c[ti + tj*blockDim.x]) *ddx* ddx*2;
real ddsdyy_num=(sc_vs*s_v[ti + tj*blockDim.x]*s_v[ti + tj*blockDim.x]+sc_vn*s_v[ti + (tj+1)*blockDim.x]*s_v[ti + (tj+1)*blockDim.x]-2*v_c*v_c*sc_c[ti + tj*blockDim.x]) *ddy* ddy*2;
real ddsdzz_num=(sc_wb*s_w0[ti + tj*blockDim.x]*s_w0[ti + tj*blockDim.x]+sc_wt*s_w1[ti + tj*blockDim.x]*s_w1[ti + tj*blockDim.x]-2*w_c*w_c*sc_c[ti + tj*blockDim.x]) *ddz* ddz*2;

     //move convective term to right hand side 
      s_c[ti + tj*blockDim.x]=-(dsudx+dsvdy+dswdz);
     //Diffusion term
real diff_center=(ddsdxx+ddsdyy+ddsdzz)*DIFF;
real diff_num=  dt*(ddsdxx_num+ddsdyy_num+ddsdzz_num);
 
//s_d[ti + tj*blockDim.x]= diff_center+diff_num;
 s_d[ti + tj*blockDim.x]= ddsdxx*(DIFF+u_c*u_c*dt/2.f)+ddsdyy*(DIFF+v_c*v_c*dt/2.f)+ddsdzz*(DIFF+w_c*w_c*dt/2.f);


real sc_conv= s_c[ti + tj*blockDim.x];
real sc_diff= s_d[ti + tj*blockDim.x];

//TODO should we have s_f at t+1/2 step as well? 
//advance scalar. Take the particle volume fraction into consideration
//real rhs=(sc_diff+s_f[ti + tj*blockDim.x])/(1-p_epsp[ti + tj*blockDim.x]);
real rhs=sc_diff+s_f[ti + tj*blockDim.x];
//real rhs=s_f[ti + tj*blockDim.x];
sc_c[ti + tj*blockDim.x] +=(sc_conv+rhs)*dt;

//if(i==30&&j==30&&k==30) printf("\nsc_diff %f %f %f %f\n",sc_diff,sc_conv,s_f[ti + tj*blockDim.x],dt);
}

// copy shared memory back to global, without copying boundary ghost values
//sc stores n+1 timestep, conv and diff are n timestep!!
    if((j >= dom->Gcc._js && j < dom->Gcc._je)
     && (i >= dom->Gcc._is && i < dom->Gcc._ie)
      && (ti > 0 && ti < (blockDim.x-1))
      && (tj > 0 && tj < (blockDim.y-1))) {
      C = i + j*dom->Gcc._s1b + k*dom->Gcc._s2b;
      sc[C] = sc_c[ti + tj*blockDim.x];
      conv[C] = s_c[ti + tj*blockDim.x];
      diff[C] = s_d[ti + tj*blockDim.x];
    }
  }
}

//first order upwind
__global__ void advance_sc_upwind_1st(real DIFF, real *u, real *v, real *w,real *f,real *epsp,
  real *diff0, real *conv0, real *diff, real *conv, real *sc,real *sc0,
  dom_struct *dom, real dt0, real dt)
{
  // create shared memory
  // no reason to load pressure into shared memory, but leaving it in global
  // will require additional if statements, so keep it in shared
  __shared__ real s_w0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w back
  __shared__ real s_w1[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w forward
  __shared__ real s_u[MAX_THREADS_DIM * MAX_THREADS_DIM];       // u
  __shared__ real s_v[MAX_THREADS_DIM * MAX_THREADS_DIM];       // v
//store scalar related values such as convective and diffusive term.
  __shared__ real s_d0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // diff0
  __shared__ real s_d[MAX_THREADS_DIM * MAX_THREADS_DIM];       // diff
  __shared__ real s_c0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // conv0
  __shared__ real s_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // conv
  __shared__ real s_f[MAX_THREADS_DIM * MAX_THREADS_DIM];       // source term


//store scalar value of difference direction at cell center
  __shared__ real sc_b[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at bottom center
  __shared__ real sc_t[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at top	 center
  __shared__ real sc_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at  	 center

//store point volume percentage in each grid cell
  __shared__ real p_epsp[MAX_THREADS_DIM * MAX_THREADS_DIM];       
 

  // working constants
  real ab0 = 0.5 * dt / dt0;  // for Adams-Bashforth stepping
  real ab = 1 + ab0;          // for Adams-Bashforth stepping

  real ddx = 1. / dom->dx;     // to limit the number of divisions needed
  real ddy = 1. / dom->dy;     // to limit the number of divisions needed
  real ddz = 1. / dom->dz;     // to limit the number of divisions needed

  

  int C;

  // loop over z-planes
//  for(int k = dom->Gcc._ksb; k < dom->Gcc._keb; k++) {
  for(int k = dom->Gcc._ks; k < dom->Gcc._ke; k++) {
    // subdomain indices
    // the extra 2*blockIdx.X terms implement the necessary overlapping of
    // shared memory blocks in the subdomain
    int i = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;

    // shared memory indices
    int ti = threadIdx.x;
    int tj = threadIdx.y;

    // load shared memory
    // TODO: look into the effect of removing these if statements and simply
    // allowing memory overruns for threads that don't matter for particular
    // discretizations

    if((j >= dom->Gcc._jsb && j < dom->Gcc._jeb)&& (i >= dom->Gcc._isb && i < dom->Gcc._ieb)) {
sc_c[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
sc_b[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k-1)*dom->Gcc._s2b];
sc_t[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k+1)*dom->Gcc._s2b];

s_c0[ti + tj*blockDim.x]=conv0[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
s_d0[ti + tj*blockDim.x]=diff0[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
 s_f[ti + tj*blockDim.x]=    f[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];

p_epsp[ti + tj*blockDim.x]= epsp[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
}

   // make sure all threads complete shared memory copy
    __syncthreads();
 
    if((j >= dom->Gfz._jsb && j < dom->Gfz._jeb)
      && (i >= dom->Gfz._isb && i < dom->Gfz._ieb)) {
      s_w0[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + k*dom->Gfz._s2b];
      s_w1[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + (k+1)*dom->Gfz._s2b];
   }
    if((j >= dom->Gfx._jsb && j < dom->Gfx._jeb)
      && (i >= dom->Gfx._isb && i < dom->Gfx._ieb)) {
      s_u[ti + tj*blockDim.x] = u[i + j*dom->Gfx._s1b + k*dom->Gfx._s2b];
    }
    if((j >= dom->Gfy._jsb && j < dom->Gfy._jeb)
      && (i >= dom->Gfy._isb && i < dom->Gfy._ieb)) {
      s_v[ti + tj*blockDim.x] = v[i + j*dom->Gfy._s1b + k*dom->Gfy._s2b];
    }
 
 
    // make sure all threads complete shared memory copy
    __syncthreads();

    // compute convective term
    // if off the shared memory block boundary
    if((ti > 0 && ti < blockDim.x-1) && (tj > 0 && tj < blockDim.y-1)) 
 
//if((j >= dom->Gcc._jsb && j < dom->Gcc._jeb)&& (i >= dom->Gcc._isb && i < dom->Gcc._ieb)) 
{
//scalar on the west and east u face
real u_c=(s_u[ti + tj*blockDim.x]+s_u[(ti+1) + tj*blockDim.x])/2;// at xm(i)
real v_c=(s_v[ti + tj*blockDim.x]+s_v[ti + (tj+1)*blockDim.x])/2;
real w_c=(s_w1[ti + tj*blockDim.x]+s_w0[ti + tj*blockDim.x])/2;
 
real sc_uw=(sc_c[ti + tj*blockDim.x]+sc_c[ti-1 + tj*blockDim.x])/2.0;//at x(i)
real sc_ue=(sc_c[ti + tj*blockDim.x]+sc_c[ti+1 + tj*blockDim.x])/2.0;

real sc_vs=(sc_c[ti + tj*blockDim.x]+sc_c[ti + (tj-1)*blockDim.x])/2.0;
real sc_vn=(sc_c[ti + tj*blockDim.x]+sc_c[ti + (tj+1)*blockDim.x])/2.0;

real sc_wb=(sc_b[ti + tj*blockDim.x]+sc_c[ti + tj*blockDim.x])/2.0;
real sc_wt=(sc_t[ti + tj*blockDim.x]+sc_c[ti + tj*blockDim.x])/2.0;

//1st order upwind scheme
real dsudx,dsvdy,dswdz;
if(u_c>0)	dsudx= (sc_c[ti + tj*blockDim.x]*u_c-sc_uw*s_u[ti + tj*blockDim.x]) *2*ddx;//backward
else		dsudx=-(sc_c[ti + tj*blockDim.x]*u_c-sc_ue*s_u[ti+1 + tj*blockDim.x]) *2* ddx;//forward

if(v_c>0)	dsvdy= (sc_c[ti + tj*blockDim.x]*v_c-sc_vs*s_v[ti + tj*blockDim.x])*2*ddy;//backward
else		dsvdy=-(sc_c[ti + tj*blockDim.x]*v_c-sc_vn*s_v[ti + (tj+1)*blockDim.x])*2*ddy;//forward

if(w_c>0)	dswdz= (sc_c[ti + tj*blockDim.x]*w_c-sc_wb*s_w0[ti + tj*blockDim.x])*2*ddz;//backward
else		dswdz=-(sc_c[ti + tj*blockDim.x]*w_c-sc_wt*s_w1[ti + tj*blockDim.x])*2*ddz;//forward


//diffusive term of scalar at cell center
real ddsdxx=(sc_c[ti-1 + tj*blockDim.x]+sc_c[ti+1 + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddx* ddx;
real ddsdyy=(sc_c[ti + (tj-1)*blockDim.x]+sc_c[ti + (tj+1)*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddy* ddy;
real ddsdzz=(sc_b[ti + tj*blockDim.x]+sc_t[ti + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddz* ddz;

     //move convective term to right hand side 
//real sc_conv=-(dsudx+dsvdy+dswdz);
      s_c[ti + tj*blockDim.x]=-(dsudx+dsvdy+dswdz);
     //Diffusion term
//real sc_diff=(ddsdxx+ddsdyy+ddsdzz)*DIFF;
      s_d[ti + tj*blockDim.x]= (ddsdxx+ddsdyy+ddsdzz)*DIFF;

//adam-bashforth at t+1/2 step
real sc_conv= ab * s_c[ti + tj*blockDim.x]-ab0*s_c0[ti + tj*blockDim.x];
real sc_diff= ab * s_d[ti + tj*blockDim.x]-ab0*s_d0[ti + tj*blockDim.x];

//TODO should we have s_f at t+1/2 step as well???
//right hand side of scalar equation
//s_rhs[ti + tj*blockDim.x]=sc_conv+sc_diff+s_f[ti + tj*blockDim.x];
//advance scalar
//Take the particle volume fraction into consideration
//real rhs=(sc_diff+s_f[ti + tj*blockDim.x])/(1-p_epsp[ti + tj*blockDim.x]);
real rhs=sc_diff+s_f[ti + tj*blockDim.x];
sc_c[ti + tj*blockDim.x] +=(sc_conv+rhs)*dt;
}

    // copy shared memory back to global, without copying boundary ghost values
//sc stores n+1 timestep, conv and diff are n timestep!!
    if((j >= dom->Gcc._js && j < dom->Gcc._je)
     && (i >= dom->Gcc._is && i < dom->Gcc._ie)
  //  if((j >= dom->Gcc._jsb && j < dom->Gcc._jeb)
  //    && (i >= dom->Gcc._isb && i < dom->Gcc._ieb)
      && (ti > 0 && ti < (blockDim.x-1))
      && (tj > 0 && tj < (blockDim.y-1))) {
      C = i + j*dom->Gcc._s1b + k*dom->Gcc._s2b;
      sc[C] = sc_c[ti + tj*blockDim.x];
      conv[C] = s_c[ti + tj*blockDim.x];
      diff[C] = s_d[ti + tj*blockDim.x];
    }
  }
}


//first order upwind
__global__ void advance_sc_upwind_1st_init(real DIFF, real *u, real *v, real *w,real *f,real *epsp,
  real *diff0, real *conv0, real *diff, real *conv, real *sc,real *sc0,
  dom_struct *dom, real dt0, real dt)
{
  // create shared memory
  // no reason to load pressure into shared memory, but leaving it in global
  // will require additional if statements, so keep it in shared
  __shared__ real s_w0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w back
  __shared__ real s_w1[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w forward
  __shared__ real s_u[MAX_THREADS_DIM * MAX_THREADS_DIM];       // u
  __shared__ real s_v[MAX_THREADS_DIM * MAX_THREADS_DIM];       // v
//store scalar related values such as convective and diffusive term.
  __shared__ real s_d[MAX_THREADS_DIM * MAX_THREADS_DIM];       // diff
  __shared__ real s_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // conv
  __shared__ real s_f[MAX_THREADS_DIM * MAX_THREADS_DIM];       // source term


//store scalar value of difference direction at cell center
  __shared__ real sc_b[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at bottom center
  __shared__ real sc_t[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at top	 center
  __shared__ real sc_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at  	 center

//store point volume percentage in each grid cell
  __shared__ real p_epsp[MAX_THREADS_DIM * MAX_THREADS_DIM];       
 

 
  real ddx = 1. / dom->dx;     // to limit the number of divisions needed
  real ddy = 1. / dom->dy;     // to limit the number of divisions needed
  real ddz = 1. / dom->dz;     // to limit the number of divisions needed

 
 

  int C;

  // loop over z-planes
//  for(int k = dom->Gcc._ksb; k < dom->Gcc._keb; k++) {
  for(int k = dom->Gcc._ks; k < dom->Gcc._ke; k++) {
    // subdomain indices
    // the extra 2*blockIdx.X terms implement the necessary overlapping of
    // shared memory blocks in the subdomain
    int i = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;

    // shared memory indices
    int ti = threadIdx.x;
    int tj = threadIdx.y;

    // load shared memory
    // TODO: look into the effect of removing these if statements and simply
    // allowing memory overruns for threads that don't matter for particular
    // discretizations

    if((j >= dom->Gcc._jsb && j < dom->Gcc._jeb)&& (i >= dom->Gcc._isb && i < dom->Gcc._ieb)) {
sc_c[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
sc_b[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k-1)*dom->Gcc._s2b];
sc_t[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k+1)*dom->Gcc._s2b];

/*
s_c0[ti + tj*blockDim.x]=conv0[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
s_d0[ti + tj*blockDim.x]=diff0[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
*/
s_f[ti + tj*blockDim.x]=    f[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];

p_epsp[ti + tj*blockDim.x]= epsp[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
}

   // make sure all threads complete shared memory copy
    __syncthreads();
 
    if((j >= dom->Gfz._jsb && j < dom->Gfz._jeb)
      && (i >= dom->Gfz._isb && i < dom->Gfz._ieb)) {
      s_w0[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + k*dom->Gfz._s2b];
      s_w1[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + (k+1)*dom->Gfz._s2b];
   }
    if((j >= dom->Gfx._jsb && j < dom->Gfx._jeb)
      && (i >= dom->Gfx._isb && i < dom->Gfx._ieb)) {
      s_u[ti + tj*blockDim.x] = u[i + j*dom->Gfx._s1b + k*dom->Gfx._s2b];
    }
    if((j >= dom->Gfy._jsb && j < dom->Gfy._jeb)
      && (i >= dom->Gfy._isb && i < dom->Gfy._ieb)) {
      s_v[ti + tj*blockDim.x] = v[i + j*dom->Gfy._s1b + k*dom->Gfy._s2b];
    }
 
 
    // make sure all threads complete shared memory copy
    __syncthreads();

    // compute convective term
    // if off the shared memory block boundary
    if((ti > 0 && ti < blockDim.x-1) && (tj > 0 && tj < blockDim.y-1)) 
 
//if((j >= dom->Gcc._jsb && j < dom->Gcc._jeb)&& (i >= dom->Gcc._isb && i < dom->Gcc._ieb)) 
{
//scalar on the west and east u face
real u_c=(s_u[ti + tj*blockDim.x]+s_u[(ti+1) + tj*blockDim.x])/2;// at xm(i)
real v_c=(s_v[ti + tj*blockDim.x]+s_v[ti + (tj+1)*blockDim.x])/2;
real w_c=(s_w1[ti + tj*blockDim.x]+s_w0[ti + tj*blockDim.x])/2;
 
real sc_uw=(sc_c[ti + tj*blockDim.x]+sc_c[ti-1 + tj*blockDim.x])/2.0;//at x(i)
real sc_ue=(sc_c[ti + tj*blockDim.x]+sc_c[ti+1 + tj*blockDim.x])/2.0;

real sc_vs=(sc_c[ti + tj*blockDim.x]+sc_c[ti + (tj-1)*blockDim.x])/2.0;
real sc_vn=(sc_c[ti + tj*blockDim.x]+sc_c[ti + (tj+1)*blockDim.x])/2.0;

real sc_wb=(sc_b[ti + tj*blockDim.x]+sc_c[ti + tj*blockDim.x])/2.0;
real sc_wt=(sc_t[ti + tj*blockDim.x]+sc_c[ti + tj*blockDim.x])/2.0;

//1st order upwind scheme
real dsudx,dsvdy,dswdz;
if(u_c>0)	dsudx= (sc_c[ti + tj*blockDim.x]*u_c-sc_uw*s_u[ti + tj*blockDim.x]) *2*ddx;//backward
else		dsudx=-(sc_c[ti + tj*blockDim.x]*u_c-sc_ue*s_u[ti+1 + tj*blockDim.x]) *2* ddx;//forward

if(v_c>0)	dsvdy= (sc_c[ti + tj*blockDim.x]*v_c-sc_vs*s_v[ti + tj*blockDim.x])*2*ddy;//backward
else		dsvdy=-(sc_c[ti + tj*blockDim.x]*v_c-sc_vn*s_v[ti + (tj+1)*blockDim.x])*2*ddy;//forward

if(w_c>0)	dswdz= (sc_c[ti + tj*blockDim.x]*w_c-sc_wb*s_w0[ti + tj*blockDim.x])*2*ddz;//backward
else		dswdz=-(sc_c[ti + tj*blockDim.x]*w_c-sc_wt*s_w1[ti + tj*blockDim.x])*2*ddz;//forward


//diffusive term of scalar at cell center
real ddsdxx=(sc_c[ti-1 + tj*blockDim.x]+sc_c[ti+1 + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddx* ddx;
real ddsdyy=(sc_c[ti + (tj-1)*blockDim.x]+sc_c[ti + (tj+1)*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddy* ddy;
real ddsdzz=(sc_b[ti + tj*blockDim.x]+sc_t[ti + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddz* ddz;

     //move convective term to right hand side 
//real sc_conv=-(dsudx+dsvdy+dswdz);
      s_c[ti + tj*blockDim.x]=-(dsudx+dsvdy+dswdz);
     //Diffusion term
//real sc_diff=(ddsdxx+ddsdyy+ddsdzz)*DIFF;
      s_d[ti + tj*blockDim.x]= (ddsdxx+ddsdyy+ddsdzz)*DIFF;

//adam-bashforth at t+1/2 step
real sc_conv=  s_c[ti + tj*blockDim.x] ;
real sc_diff=  s_d[ti + tj*blockDim.x] ;

//TODO should we have s_f at t+1/2 step as well???
//right hand side of scalar equation
//s_rhs[ti + tj*blockDim.x]=sc_conv+sc_diff+s_f[ti + tj*blockDim.x];
//advance scalar
//Take the particle volume fraction into consideration
//real rhs=(sc_diff+s_f[ti + tj*blockDim.x])/(1-p_epsp[ti + tj*blockDim.x]);
real rhs=sc_diff+s_f[ti + tj*blockDim.x];
sc_c[ti + tj*blockDim.x] +=(sc_conv+rhs)*dt;
}

    // copy shared memory back to global, without copying boundary ghost values
//sc stores n+1 timestep, conv and diff are n timestep!!
    if((j >= dom->Gcc._js && j < dom->Gcc._je)
     && (i >= dom->Gcc._is && i < dom->Gcc._ie)
  //  if((j >= dom->Gcc._jsb && j < dom->Gcc._jeb)
  //    && (i >= dom->Gcc._isb && i < dom->Gcc._ieb)
      && (ti > 0 && ti < (blockDim.x-1))
      && (tj > 0 && tj < (blockDim.y-1))) {
      C = i + j*dom->Gcc._s1b + k*dom->Gcc._s2b;
      sc[C] = sc_c[ti + tj*blockDim.x];
      conv[C] = s_c[ti + tj*blockDim.x];
      diff[C] = s_d[ti + tj*blockDim.x];
    }
  }
}


//Using central difference scheme in space, and adam-bashforth scheme in time.
__global__ void advance_sc(	real DIFF, 
				real *u, real *v, real *w,
				real *f,real *epsp,
				real *diff0, real *conv0, 
				real *diff, real *conv, 
				real *sc,real *sc0,
				dom_struct *dom, real dt0, real dt)
{
  // create shared memory
  // no reason to load pressure into shared memory, but leaving it in global
  // will require additional if statements, so keep it in shared
  __shared__ real s_w0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w back
  __shared__ real s_w1[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w center
  __shared__ real s_u[MAX_THREADS_DIM * MAX_THREADS_DIM];       // u
  __shared__ real s_v[MAX_THREADS_DIM * MAX_THREADS_DIM];       // v
//store scalar related values such as convective and diffusive term.
  __shared__ real s_d0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // diff0
  __shared__ real s_d[MAX_THREADS_DIM * MAX_THREADS_DIM];       // diff
  __shared__ real s_c0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // conv0
  __shared__ real s_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // conv
  __shared__ real s_f[MAX_THREADS_DIM * MAX_THREADS_DIM];       // source term


//store scalar value of difference direction at cell center
  __shared__ real sc_b[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at bottom center
  __shared__ real sc_t[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at top	 center
  __shared__ real sc_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at  	 center

//store point volume percentage in each grid cell
  __shared__ real p_epsp[MAX_THREADS_DIM * MAX_THREADS_DIM];       
 

  // working constants
  real ab0 = 0.5 * dt / dt0;  // for Adams-Bashforth stepping
  real ab = 1 + ab0;          // for Adams-Bashforth stepping

  real ddx = 1. / dom->dx;     // to limit the number of divisions needed
  real ddy = 1. / dom->dy;     // to limit the number of divisions needed
  real ddz = 1. / dom->dz;     // to limit the number of divisions needed

 
  int C;

  // loop over z-planes
//  for(int k = dom->Gcc._ksb; k < dom->Gcc._keb; k++) {
  for(int k = dom->Gcc._ks; k < dom->Gcc._ke; k++) {
    // subdomain indices
    // the extra 2*blockIdx.X terms implement the necessary overlapping of
    // shared memory blocks in the subdomain
    int i = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;

    // shared memory indices
    int ti = threadIdx.x;
    int tj = threadIdx.y;

    // load shared memory
    // TODO: look into the effect of removing these if statements and simply
    // allowing memory overruns for threads that don't matter for particular
    // discretizations

    if((j >= dom->Gcc._jsb && j < dom->Gcc._jeb)&& (i >= dom->Gcc._isb && i < dom->Gcc._ieb)) {
sc_c[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
sc_b[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k-1)*dom->Gcc._s2b];
sc_t[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k+1)*dom->Gcc._s2b];

s_c0[ti + tj*blockDim.x]=conv0[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
s_d0[ti + tj*blockDim.x]=diff0[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
 s_f[ti + tj*blockDim.x]=    f[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];

p_epsp[ti + tj*blockDim.x]= epsp[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];

//if(isnan(sc_c[ti + tj*blockDim.x])) printf("\nglobal_var %f %f %f %f %d %d %d %d %d\n",sc_b[ti + tj*blockDim.x],sc_t[ti + tj*blockDim.x],s_f[ti + tj*blockDim.x],sc_c[ti + tj*blockDim.x],ti,tj,i,j,k);

}

   // make sure all threads complete shared memory copy
    __syncthreads();

    if((j >= dom->Gfz._js && j < dom->Gfz._je)
      && (i >= dom->Gfz._is && i < dom->Gfz._ie)) {
      s_w0[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + k*dom->Gfz._s2b];
      s_w1[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + (k+1)*dom->Gfz._s2b];
    }
    if((j >= dom->Gfx._js && j < dom->Gfx._je)
      && (i >= dom->Gfx._is && i < dom->Gfx._ie)) {
      s_u[ti + tj*blockDim.x] = u[i + j*dom->Gfx._s1b + k*dom->Gfx._s2b];
    }
    if((j >= dom->Gfy._js && j < dom->Gfy._je)
      && (i >= dom->Gfy._is && i < dom->Gfy._ie)) {
      s_v[ti + tj*blockDim.x] = v[i + j*dom->Gfy._s1b + k*dom->Gfy._s2b];
    }

  
    // make sure all threads complete shared memory copy
    __syncthreads();

    // compute convective term
    // if off the shared memory block boundary
    if((ti > 0 && ti < blockDim.x-1) && (tj > 0 && tj < blockDim.y-1)) {
//scalar at face center
real sc_uw=(sc_c[ti + tj*blockDim.x]+sc_c[ti-1 + tj*blockDim.x])/2.0f;
real sc_ue=(sc_c[ti + tj*blockDim.x]+sc_c[ti+1 + tj*blockDim.x])/2.0f;

real sc_vs=(sc_c[ti + tj*blockDim.x]+sc_c[ti + (tj-1)*blockDim.x])/2.0f;
real sc_vn=(sc_c[ti + tj*blockDim.x]+sc_c[ti + (tj+1)*blockDim.x])/2.0f;

real sc_wb=(sc_b[ti + tj*blockDim.x]+sc_c[ti + tj*blockDim.x])/2.0f;
real sc_wt=(sc_t[ti + tj*blockDim.x]+sc_c[ti + tj*blockDim.x])/2.0f;

//convective term of scalar at cell center
real dsudx=(s_u[(ti+1) + tj*blockDim.x]*sc_ue - s_u[ti + tj*blockDim.x]*sc_uw) * ddx;
real dsvdy=(s_v[ti + (tj+1)*blockDim.x]*sc_vn - s_v[ti + tj*blockDim.x]*sc_vs) * ddy;
real dswdz=(s_w1[ti + tj*blockDim.x]   *sc_wt - s_w0[ti + tj*blockDim.x]*sc_wb) * ddz;

//diffusive term of scalar at cell center
real ddsdxx=(sc_c[ti-1 + tj*blockDim.x]+sc_c[ti+1 + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddx* ddx;
real ddsdyy=(sc_c[ti + (tj-1)*blockDim.x]+sc_c[ti + (tj+1)*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddy* ddy;
real ddsdzz=(sc_b[ti + tj*blockDim.x]+sc_t[ti + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddz* ddz;

     //move convective term to right hand side 
      s_c[ti + tj*blockDim.x]=-(dsudx+dsvdy+dswdz);
     //Diffusion term
      s_d[ti + tj*blockDim.x]= (ddsdxx+ddsdyy+ddsdzz)*DIFF;

//if(isnan(sc_c[ti + tj*blockDim.x])) printf("\nconv_diff %f %f %f %f %f %d %d %d %d %d\n",dsudx,ddsdxx,ddx,ab,sc_c[ti + tj*blockDim.x],ti,tj,i,j,k);
//Adam-bashforth at t+1/2 step
real sc_conv= ab * s_c[ti + tj*blockDim.x]-ab0*s_c0[ti + tj*blockDim.x];
real sc_diff= ab * s_d[ti + tj*blockDim.x]-ab0*s_d0[ti + tj*blockDim.x];

//if(isnan(s_f[ti + tj*blockDim.x])) printf("\nsource %f %f %f %f %d %d %d %d %d\n",s_d[ti + tj*blockDim.x],s_c[ti + tj*blockDim.x],s_f[ti + tj*blockDim.x],sc_c[ti + tj*blockDim.x],ti,tj,i,j,k);

//TODO should we have s_f at t+1/2 step as well? 
//advance scalar. Take the particle volume fraction into consideration
//real rhs=(sc_diff+s_f[ti + tj*blockDim.x])/(1-p_epsp[ti + tj*blockDim.x]);
real rhs=sc_diff+s_f[ti + tj*blockDim.x];
//real rhs=s_f[ti + tj*blockDim.x];
sc_c[ti + tj*blockDim.x] +=(sc_conv+rhs)*dt;

//if(i==36&&j==38&&k==35) printf("\nsc_diff %f %f %f %f\n",sc_diff,sc_conv,s_f[ti + tj*blockDim.x],sc_c[ti + tj*blockDim.x]);
}

// copy shared memory back to global, without copying boundary ghost values
//sc stores n+1 timestep, conv and diff are n timestep!!
    if((j >= dom->Gcc._js && j < dom->Gcc._je)
     && (i >= dom->Gcc._is && i < dom->Gcc._ie)
      && (ti > 0 && ti < (blockDim.x-1))
      && (tj > 0 && tj < (blockDim.y-1))) {
      C = i + j*dom->Gcc._s1b + k*dom->Gcc._s2b;
//if(isnan(sc_c[ti + tj*blockDim.x])) printf("\ntest_diff %f %f %f %f %d %d %d %d %d\n",s_d[ti + tj*blockDim.x],s_c[ti + tj*blockDim.x],s_f[ti + tj*blockDim.x],sc_c[ti + tj*blockDim.x],ti,tj,i,j,k);
      sc[C] = sc_c[ti + tj*blockDim.x];
      conv[C] = s_c[ti + tj*blockDim.x];
      diff[C] = s_d[ti + tj*blockDim.x];
    }
  }
}




__global__ void advance_sc_init(real DIFF, real *u, real *v, real *w,real *f,real *epsp,
  real *diff0, real *conv0, real *diff, real *conv, real *sc,real *sc0,
  dom_struct *dom, real dt0, real dt)
{
  // create shared memory
  // no reason to load pressure into shared memory, but leaving it in global
  // will require additional if statements, so keep it in shared
  __shared__ real s_w0[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w back
  __shared__ real s_w1[MAX_THREADS_DIM * MAX_THREADS_DIM];      // w center
  __shared__ real s_u[MAX_THREADS_DIM * MAX_THREADS_DIM];       // u
  __shared__ real s_v[MAX_THREADS_DIM * MAX_THREADS_DIM];       // v
  
//store scalar related values such as convective and diffusive term.
  __shared__ real s_d[MAX_THREADS_DIM * MAX_THREADS_DIM];       // diff
//  __shared__ real s_d0[MAX_THREADS_DIM * MAX_THREADS_DIM];       // diff0
  __shared__ real s_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // conv
//  __shared__ real s_c0[MAX_THREADS_DIM * MAX_THREADS_DIM];       // conv0
  __shared__ real s_f[MAX_THREADS_DIM * MAX_THREADS_DIM];       // source term


//store scalar value of difference direction at cell center
  __shared__ real sc_c[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at  	 center
  __shared__ real sc_b[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at bottom center
  __shared__ real sc_t[MAX_THREADS_DIM * MAX_THREADS_DIM];       // sc at top	 center

//store point volume percentage in each grid cell
  __shared__ real p_epsp[MAX_THREADS_DIM * MAX_THREADS_DIM];       

  // working constants
  real ddx = 1. / dom->dx;     // to limit the number of divisions needed
  real ddy = 1. / dom->dy;     // to limit the number of divisions needed
  real ddz = 1. / dom->dz;     // to limit the number of divisions needed

  int C;

  // loop over z-planes
//  for(int k = dom->Gcc._ksb; k < dom->Gcc._keb; k++) {
  for(int k = dom->Gcc._ks; k < dom->Gcc._ke; k++) {
    // subdomain indices
    // the extra 2*blockIdx.X terms implement the necessary overlapping of
    // shared memory blocks in the subdomain
    int i = blockIdx.x*blockDim.x + threadIdx.x - 2*blockIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y - 2*blockIdx.y;
    // shared memory indices
    int ti = threadIdx.x;
    int tj = threadIdx.y;
 
    // load shared memory
    // TODO: look into the effect of removing these if statements and simply
    // allowing memory overruns for threads that don't matter for particular
    // discretizations

    if((j >= dom->Gcc._jsb && j < dom->Gcc._jeb)&& (i >= dom->Gcc._isb && i < dom->Gcc._ieb)) {
sc_c[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
sc_b[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k-1)*dom->Gcc._s2b];
sc_t[ti + tj*blockDim.x]=sc0[i+j*dom->Gcc._s1b + (k+1)*dom->Gcc._s2b];

p_epsp[ti + tj*blockDim.x]= epsp[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];

s_f[ti + tj*blockDim.x]=    f[i+j*dom->Gcc._s1b + k*dom->Gcc._s2b];
}

  // make sure all threads complete shared memory copy
    __syncthreads();

    if((j >= dom->Gfz._js && j < dom->Gfz._je)
      && (i >= dom->Gfz._is && i < dom->Gfz._ie)) {
      s_w0[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + k*dom->Gfz._s2b];
      s_w1[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + (k+1)*dom->Gfz._s2b];
    }
    if((j >= dom->Gfx._js && j < dom->Gfx._je)
      && (i >= dom->Gfx._is && i < dom->Gfx._ie)) {
      s_u[ti + tj*blockDim.x] = u[i + j*dom->Gfx._s1b + k*dom->Gfx._s2b];
    }
    if((j >= dom->Gfy._js && j < dom->Gfy._je)
      && (i >= dom->Gfy._is && i < dom->Gfy._ie)) {
      s_v[ti + tj*blockDim.x] = v[i + j*dom->Gfy._s1b + k*dom->Gfy._s2b];
    }

/*
   if((j >= dom->Gfz._jsb && j < dom->Gfz._jeb)
      && (i >= dom->Gfz._isb && i < dom->Gfz._ieb)) {
      s_w0[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + k*dom->Gfz._s2b];
      s_w1[ti + tj*blockDim.x] = w[i + j*dom->Gfz._s1b + (k+1)*dom->Gfz._s2b];
    }
    if((j >= dom->Gfx._jsb && j < dom->Gfx._jeb)
      && (i >= dom->Gfx._isb && i < dom->Gfx._ieb)) {
      s_u[ti + tj*blockDim.x] = u[i + j*dom->Gfx._s1b + k*dom->Gfx._s2b];
    }
    if((j >= dom->Gfy._jsb && j < dom->Gfy._jeb)
      && (i >= dom->Gfy._isb && i < dom->Gfy._ieb)) {
      s_v[ti + tj*blockDim.x] = v[i + j*dom->Gfy._s1b + k*dom->Gfy._s2b];
    }
*/


/*
    //convective term
    s_c[ti + tj*blockDim.x] = 0.0;
    //diffusion term
    s_d[ti + tj*blockDim.x] = 0.0;
*/
   // make sure all threads complete shared memory copy
    __syncthreads();
 
    // compute convective term
    // if off the shared memory block boundary
    if((ti > 0 && ti < blockDim.x-1) && (tj > 0 && tj < blockDim.y-1)) {
//scalar on the west and east u face
real sc_uw=(sc_c[ti + tj*blockDim.x]+sc_c[ti-1 + tj*blockDim.x])/2.0f;
real sc_ue=(sc_c[ti + tj*blockDim.x]+sc_c[ti+1 + tj*blockDim.x])/2.0f;

real sc_vs=(sc_c[ti + tj*blockDim.x]+sc_c[ti + (tj-1)*blockDim.x])/2.0f;
real sc_vn=(sc_c[ti + tj*blockDim.x]+sc_c[ti + (tj+1)*blockDim.x])/2.0f;

real sc_wb=(sc_b[ti + tj*blockDim.x]+sc_c[ti + tj*blockDim.x])/2.0f;
real sc_wt=(sc_t[ti + tj*blockDim.x]+sc_c[ti + tj*blockDim.x])/2.0f;

//convective term of scalar at cell center
real dsudx=(s_u[(ti+1) + tj*blockDim.x]*sc_ue - s_u[ti + tj*blockDim.x]*sc_uw) * ddx;
real dsvdy=(s_v[ti + (tj+1)*blockDim.x]*sc_vn - s_v[ti + tj*blockDim.x]*sc_vs) * ddy;
real dswdz=(s_w1[ti + tj*blockDim.x]   *sc_wt - s_w0[ti + tj*blockDim.x]*sc_wb) * ddz;


//diffusive term of scalar at cell center
real ddsdxx=(sc_c[ti-1 + tj*blockDim.x]+sc_c[ti+1 + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddx* ddx;
real ddsdyy=(sc_c[ti + (tj-1)*blockDim.x]+sc_c[ti + (tj+1)*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddy* ddy;
real ddsdzz=(sc_b[ti + tj*blockDim.x]+sc_t[ti + tj*blockDim.x]-2*sc_c[ti + tj*blockDim.x]) *ddz* ddz;
 
     //move convective term to right hand side 
      s_c[ti + tj*blockDim.x]=-(dsudx+dsvdy+dswdz);
     //Diffusion term
      s_d[ti + tj*blockDim.x]= (ddsdxx+ddsdyy+ddsdzz)*DIFF;

//advance scalar
//sc_c[ti + tj*blockDim.x] +=(s_c[ti + tj*blockDim.x]+s_d[ti + tj*blockDim.x]+s_f[ti + tj*blockDim.x])*dt;
//real rhs=(s_d[ti + tj*blockDim.x]+s_f[ti + tj*blockDim.x])/(1-p_epsp[ti + tj*blockDim.x]);
real rhs=s_d[ti + tj*blockDim.x]+s_f[ti + tj*blockDim.x];
//real rhs=s_f[ti + tj*blockDim.x];
sc_c[ti + tj*blockDim.x] +=(s_c[ti + tj*blockDim.x]+rhs)*dt;

//if(i==36&&j==38&&k==35) printf("\nsc_diff0 %f %f %f %d\n",s_d[ti + tj*blockDim.x],s_c[ti + tj*blockDim.x],s_f[ti + tj*blockDim.x],k);

}

    __syncthreads();

    // copy shared memory back to global
//sc stores n+1 timestep, conv and diff are n timestep
  if((j >= dom->Gcc._js && j < dom->Gcc._je)
      && (i >= dom->Gcc._is && i < dom->Gcc._ie)
      && (ti > 0 && ti < (blockDim.x-1))
      && (tj > 0 && tj < (blockDim.y-1))) {
      C = i + j*dom->Gcc._s1b + k*dom->Gcc._s2b;
      sc[C] = sc_c[ti + tj*blockDim.x];
      conv[C] = s_c[ti + tj*blockDim.x];
      diff[C] = s_d[ti + tj*blockDim.x];
    }

  }
}


 __global__ void lpt_scalar_source_init(real *A, real *B,dom_struct *dom)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y;
  int ti = blockDim.z*blockIdx.z + threadIdx.z;
/*
  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;
*/
int s1b=tex1Dfetch(texRefDomInfo,21);
int s2b=tex1Dfetch(texRefDomInfo,22);

int ieb=tex1Dfetch(texRefDomInfo,4);
int jeb=tex1Dfetch(texRefDomInfo,10);
int keb=tex1Dfetch(texRefDomInfo,16);


/*
if(tj < dom->Gcc._jeb && tk < dom->Gcc._keb) {
    for(int i = dom->Gcc._isb; i < dom->Gcc._ieb; i++) 
*/

if(ti<ieb &&tj < jeb && tk < keb) {
	{
	A[ti+tj*s1b+tk*s2b]=0.f;
	B[ti+tj*s1b+tk*s2b]=0.f;
	}
 }

}





__global__ void lpt_epsp_clip(real *epsp,dom_struct *dom)
{

  int tj = blockDim.x*blockIdx.x + threadIdx.x- 2*blockIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y- 2*blockIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;
if(tj < dom->Gcc._jeb && tk < dom->Gcc._keb) {
    for(int i = dom->Gcc._isb; i < dom->Gcc._ieb; i++) 
        {
        if(epsp[i+tj*s1b+tk*s2b]>EPSP_CLIP)  epsp[i+tj*s1b+tk*s2b]=EPSP_CLIP;
        }
    }
}











//This test case only works with 0 convection!!
//Test scalar source is cos(wx*x)*cos(wy*y)*(cos(wt*t)+(wx*wx+wy*wy)*DIFF*sin(wt*t)/wt); 
//The corresponding solution is cos(wx*x)*cos(wy*y)*DIFF*sin(wt*t)/wt
__global__ void lpt_scalar_source_init_test(real *src,dom_struct *dom,real t, real DIFF)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x- 2*blockIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y- 2*blockIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;
if(tj < dom->Gcc._jeb && tk < dom->Gcc._keb) {
    for(int i = dom->Gcc._isb; i < dom->Gcc._ieb; i++) 
	{
 real x = ((i-0.5) * dom->dx) + dom->xs;
 real y = ((tj-0.5) * dom->dy) + dom->ys;
 real wt=10;
 real wx=1;
 real wy=2;
	src[i+tj*s1b+tk*s2b]=DIFF*cos(wx*x)*cos(wy*y)*(cos(wt*t)+(wx*wx+wy*wy)*DIFF*sin(wt*t)/wt);	
	}
 }

}

//Need to run and modify  forcing_test_taylor<<<numBlocks, dimBlocks>>>(_f_x[dev],_f_y[dev], _dom[dev],ttime,DIFF,_points[dev],n2,n3) in cuda_bluebottle.cu
//suppose u=U0*cos(qx)cos(qy)sin(rt); v=U0*sin(qx)sin(qy)sin(rt)
//Test scalar source is cos(wx*x)*cos(wy*y)*(cos(wt*t)+(wx*wx+wy*wy)*DIFF*sin(wt*t)/wt)-1.f/wt*U0*sin(r*t)*sin(wt*t)*(wx*cos(q*x)*cos(q*y)*cos(wy*y)*sin(wx*x)+wy*cos(wx*x)*sin(q*x)*sin(q*y)*sin(wy*y)); 
//Test solution for scalar is cos(wx*x)*cos(wy*y)*sin(wt*t)
__global__ void lpt_scalar_source_convDiff_test(real *src,dom_struct *dom,real t, real DIFF)
{
  int tj = blockDim.x*blockIdx.x + threadIdx.x- 2*blockIdx.x;
  int tk = blockDim.y*blockIdx.y + threadIdx.y- 2*blockIdx.y;

  int s1b = dom->Gcc._s1b;
  int s2b = dom->Gcc._s2b;
/*
 real wt=10*2*PI;
 real wx=6;
 real wy=8;

 real q=6;
 real r=wt;
 real U0=20;
*/

 real wt=10*2*PI;
 real wx=2;
 real wy=2;

 real q=2;
 real r=wt;
 real U0=10;


if(tj < dom->Gcc._jeb && tk < dom->Gcc._keb) {
    for(int i = dom->Gcc._isb; i < dom->Gcc._ieb; i++) 
	{
 real x = ((i-0.5) * dom->dx) + dom->xs;
 real y = ((tj-0.5) * dom->dy) + dom->ys;
	src[i+tj*s1b+tk*s2b]=cos(wx*x)*cos(wy*y)*(cos(wt*t)+(wx*wx+wy*wy)*DIFF*sin(wt*t)/wt)-1.f/wt*U0*sin(r*t)*sin(wt*t)*(wx*cos(q*x)*cos(q*y)*cos(wy*y)*sin(wx*x)+wy*cos(wx*x)*sin(q*x)*sin(q*y)*sin(wy*y));	
	}
 }

}


