#Makefile 
#define variables
objects= MontecarloEstimation.o 
NVCC= nvcc               
ARCH= -arch=sm_20        #cuda compute capability
LIBS=  
execname= Montecarlo


#compile
$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS) 

MontecarloEstimation.o: MontecarloEstimation.cu
	$(NVCC) $(opt) $(ARCH)  -c -g MontecarloEstimation.cu #-lcurand -L /usr/local/cuda/lib64/stubs 


#clean Makefile
clean:
	rm $(objects) $(execname)
#end of Makefile


