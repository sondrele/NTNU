all : mc_cuda mc_cl

USER=sondrele@gpu01.idi.ntnu.no

mc_cuda: mc.cu
	nvcc -arch=sm_20 mc.cu -o mc_cuda -lcudart -lGL -lGLU -lglut -lGLEW

mc_cl: mc.c clutil.c mc.cl
	gcc -std=c99 mc.c clutil.c -o mc_cl -lGL -lGLU -lglut -lGLEW -lOpenCL

runcu: run
	ssh $(USER) 'cd ~/TDT4200/PS6; make mc_cuda; ./mc_cuda 64 256'

runcl: run
	ssh $(USER) 'cd ~/TDT4200/PS6; make mc_cl; ./mc_cl 8 1024 > kres'

run: upload
	ssh $(USER) 'cd ~/TDT4200/PS6;make unzip; make clean'

upload: zip
	scp ps6.zip $(USER):~/TDT4200/PS6/

zip:
	zip ps6.zip Makefile mc.c mc.cl mc.cu

unzip:
	unzip -o ps6.zip

clean:
	rm -f mc_cl mc_cuda
