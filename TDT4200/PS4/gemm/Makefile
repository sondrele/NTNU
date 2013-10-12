all: gemm gemm_openmp gemm_pthread

gemm: gemm.c
	gcc -std=c99 -O3 -o gemm gemm.c -lm
	
gemm_openmp: gemm_openmp.c
	gcc -std=c99 -O3 -o gemm_openmp gemm_openmp.c -lm -fopenmp

gemm_pthread: gemm_pthread.c
	gcc -std=c99 -O3 -o gemm_pthread gemm_pthread.c -lm -lpthread
	
run: all
	time ./gemm 2 2048 2048 512

runmp: all
	time ./gemm_openmp 1024 2048 2048 512

runt: all
	./gemm_pthread 1 8 8 8

clean:
	-rm -f gemm gemm_pthread gemm_openmp
