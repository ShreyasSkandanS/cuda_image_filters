CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
#CFLAGS += -O2 -s -DNDEBUG

all:
	nvcc -I. -arch=sm_52 -c src/median_gpu.cu -o build/median_gpu.o 
	nvcc -I. -arch=sm_52 -c src/bilateral_gpu.cu -o build/bilateral_gpu.o
	g++ -o build/filters_gpu src/main.cpp build/median_gpu.o build/bilateral_gpu.o $(CFLAGS) $(LIBS) -L/usr/local/cuda/lib64 -lcudart

clean: 
	@rm -rf *.o build/filters_gpu build/median_gpu build/bilateral_gpu
	@rm -rf *~
