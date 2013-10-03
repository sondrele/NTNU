CFLAGS = -I. -I/share/apps/atlas/include -std=c99 -O3 -ffast-math
LDFLAGS = -L/share/apps/atlas/lib
LDLIBS = -latlas -lcblas -lm

all: gemm_naive gemm_atlas gemm_fast

gemm_naive : gemm.c naive.o
	gcc $(CFLAGS) $(LDFLAGS) gemm.c naive.o -o gemm_naive $(LDLIBS) 

gemm_atlas : gemm.c atlas.o
	gcc $(CFLAGS) $(LDFLAGS) gemm.c atlas.o -o gemm_atlas  $(LDLIBS)

gemm_fast : gemm.c fast.o
	gcc $(CFLAGS) $(LDFLAGS) -std=c99 -O3 -msse3 gemm.c fast.o -o gemm_fast $(LDLIBS)

%.o : %.c
	gcc $(CFLAGS) -msse3 -c $<

run: all
	./gemm_naive 256 256 256
	./gemm_fast 256 256 256

update:
	scp Makefile sondrele@clustis3.idi.ntnu.no:~/TDT4200/PS3/

upload:
	scp fast.c sondrele@clustis3.idi.ntnu.no:~/TDT4200/PS3/

runnaive: upload
	ssh sondrele@clustis3.idi.ntnu.no 'cd ~/TDT4200/PS3/;make run'

clean:
	rm gemm_* *.o
