PREC = DOUBLE

C = mpicc.openmpi
COPT = -std=c99 -pedantic -Wall -Wextra -fopenmp -D$(PREC)
LDINCS = -I /usr/lib/openmpi/include -I /opt/cgns/include -I /opt/cusplibrary
LDLIBS = -lm -L /opt/hdf5/lib -L /opt/cgns/lib -lcgns -lhdf5 
CUSP_DIR=/opt/cusplibrary

CUDA = nvcc
  # compiled to sm_30 because cuda-memcheck can't handle sm_35
CUDAOPT = -arch=sm_30 -Xcompiler -fopenmp -m64 -D$(PREC)
SDK_DIR = /opt/NVIDIA_CUDA-5.5_Samples
CUDA_DIR = /opt/cuda/lib64

CUDA_HEAD = /opt/cuda-5.5

CUDAPROFINCS= -I $(CUDA_HEAD)/include
CUDAINCS = -I $(SDK_DIR)/common/inc -I $(CUDA_DIR)/include
CUDALIBS = -L $(SDK_DIR)/lib -L $(CUDA_DIR) 	\
	-lcudart

SRCC =	bluebottle.c	\
	domain.c	\
	precursor.c	\
	vtk.c           \
	particle.c 	\
	recorder.c	\
	seeder.c	\
	point.c		\
	scalar.c	

SRCCUDA = cuda_bluebottle.cu	\
	cuda_bicgstab.cu	\
	cuda_testing.cu		\
	cuda_quadrature.cu	\
	cuda_particle.cu	\
	quadrature_kernel.cu	\
	entrySearch.cu		\
	bluebottle_kernel.cu	\
	bicgstab_kernel.cu	\
	entrySearch_kernel.cu	\
	particle_kernel.cu	\
	point_kernel.cu		\
	cuda_point.cu	\
	cuda_scalar.cu	\
	scalar_kernel.cu	

EXTRA = Makefile		\
	bluebottle.h		\
	cuda_bluebottle.h	\
	cuda_bicgstab.h		\
	cuda_quadrature.h	\
	cuda_particle.h		\
	particle.h		\
	cuda_testing.h		\
	domain.h		\
	vtk.h                   \
	entrySearch.h		\
	precursor.h		\
	recorder.h		\
	cuda_point.h		\
	cuda_scalar.h		\
	point.h			\
	scalar.h		
#header file to include when remake it
HEADER =scalar.h \
	point.h	\
	cuda_point.h

# compile normally
all: COPT += -O3
all: CUDAOPT += -O3
all: bluebottle

# compile for batch job submission
batch: COPT += -O3 -DBATCHRUN
batch: CUDAOPT += -O3
batch: bluebottle

# compile with stair-stepped interior boundaries
steps: COPT += -DSTEPS -O2
steps: CUDAOPT += -DSTEPS -O2
steps: bluebottle

# compile with debug output
debug: COPT += -DDEBUG -g 
debug: CUDAOPT += -DDEBUG -g -G 
debug: bluebottle

# compile with testing code
test: COPT += -DDEBUG -DTEST -g
test: CUDAOPT += -DDEBUG -DTEST -g -G
test: bluebottle

# write robodoc documentation
doc:
	cd .. && robodoc --html --multidoc --doc doc/robodoc && robodoc --latex --singledoc --sections --doc doc/LaTeX/Bluebottle_0.1_robodoc && cd doc/LaTeX && pdflatex Bluebottle_0.1_robodoc.tex && pdflatex Bluebottle_0.1_robodoc.tex && pdflatex Bluebottle_0.1_robodoc.tex && echo '\nmake doc: Complete.'

OBJS = $(addsuffix .o, $(basename $(SRCC)))
OBJSCUDA = $(addsuffix .o, $(basename $(SRCCUDA)))

%.o:%.cu $(HEADER)
	$(CUDA) $(CUDAOPT) -dc $< $(CUDAINCS) $(LDINCS)

%.o:%.c
	$(C) $(COPT) -c $< $(LDINCS) $(CUDAPROFINCS)

bblib.o: $(OBJSCUDA) 
	$(CUDA) $(CUDAOPT) -dlink $+ -o $@ $(CUDALIBS)

bluebottle: $(OBJSCUDA) bblib.o $(OBJS)
	$(C) $(COPT) -o $@ $+ $(LDLIBS) $(CUDALIBS) -lstdc++

clean:
	rm -f *.o bluebottle seeder
