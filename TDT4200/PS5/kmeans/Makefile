JPGS=$(shell ls *.dat | sed s/\.dat/.jpg/)
ARG=kres

all: kmeans kmeans_cuda

kmeans: kmeans.c
	gcc -std=c99 -O3 -o kmeans kmeans.c -lm
	
kmeans_cuda: kmeans_cuda.cu
	nvcc -arch=sm_20 -O3 -o kmeans_cuda kmeans_cuda.cu -lcudart -lm

plot: plot.txt
	gnuplot gpbatch.txt
	
plot.txt: kmeans
	cat $(ARG) > plot.txt
	
	
jpgs: ${JPGS}

%.jpg: %.dat
	gnuplot -e "infile='$*.dat'; outfile='$*.jpg'" gpbatch.txt
	-rm -f $*.dat

runcuda: run
	ssh sondrele@gpu01.idi.ntnu.no 'cd ~/TDT4200/PS5/kmeans; ./kmeans_cuda 4 64'

runserial: run
	ssh sondrele@gpu01.idi.ntnu.no 'cd ~/TDT4200/PS5/kmeans; ./kmeans 8 1024 > kres'

run: upload
	ssh sondrele@gpu01.idi.ntnu.no 'cd ~/TDT4200/PS5/kmeans;make unzip; make clean; make'

upload: zip
	scp ps5.zip sondrele@gpu01.idi.ntnu.no:~/TDT4200/PS5/kmeans/

zip:
	zip ps5.zip Makefile gpbatch.txt kmeans.c kmeans_cuda.cu compare.sh

unzip:
	unzip -o ps5.zip
	
clean:
	-rm -f kmeans kmeans_cuda plot.txt plot.jpg
