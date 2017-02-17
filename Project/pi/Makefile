#Makefile 
#define variables
objects= pi.o 
NVCC= nvcc               
ARCH= -arch=sm_20        #cuda compute capability
LIBS=  
execname= pi


#compile
$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS) 

pi.o: pi.cu
	$(NVCC) $(opt) $(ARCH)  -c -g pi.cu #-lcurand -L /usr/local/cuda/lib64/stubs 


#clean Makefile
clean:
	rm $(objects) $(execname)
#end of Makefile


