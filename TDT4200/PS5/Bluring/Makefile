CFLAGS += -std=c99

all : blur_cuda

blur_cuda: bmp.o host_blur.o blur_cuda.cu
	nvcc -arch=sm_20  host_blur.o bmp.o blur_cuda.cu -o blur_cuda -lcudart
	
clean:
	rm *.o blur_cuda

run: upload
	ssh sondrele@gpu01.idi.ntnu.no 'cd ~/TDT4200/PS5/blur;make unzip; make clean; make; ./blur_cuda'

upload: zip
	scp ps5.zip sondrele@gpu01.idi.ntnu.no:~/TDT4200/PS5/blur/

zip:
	zip ps5.zip Makefile blur_cuda.cu host_blur.c host_blur.h bmp.h bmp.c peppers.bmp

unzip:
	unzip -o ps5.zip