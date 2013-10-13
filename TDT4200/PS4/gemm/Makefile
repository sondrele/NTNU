all: gemm gemm_openmp gemm_pthread

gemm: gemm.c
	gcc -std=c99 -O3 -o gemm gemm.c -lm
	
gemm_openmp: gemm_openmp.c
	gcc -std=c99 -O3 -o gemm_openmp gemm_openmp.c -lm -fopenmp

gemm_pthread: gemm_pthread.c
	gcc -std=c99 -O3 -o gemm_pthread gemm_pthread.c -lm -lpthread
	
run: all
	time ./gemm 4 8 8 1

runmp: all
	time ./gemm_openmp 4 8 8 1

runt: all
	time ./gemm_pthread 4 8 8 1

testmp: all
	time ./gemm 512 1024 1024 512 > g_res
	time ./gemm_openmp 512 1024 1024 512 > mp_res
	diff g_res mp_res

testt: all
	./gemm 4 4 4 1 > g_res
	./gemm_pthread 4 4 4 1 > t_res
	diff g_res t_res

clean:
	-rm -f gemm gemm_pthread gemm_openmp
