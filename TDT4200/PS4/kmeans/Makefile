#JPGS=$(shell ls *.dat | sed s/\.dat/.jpg/)

T=2
C=2
P=16

all: kmeans kmeans_openmp kmeans_pthread

kmeans: kmeans.c
	gcc -std=c99 -O3 -o kmeans kmeans.c -lm
	
kmeans_openmp: kmeans_openmp.c
	gcc -std=c99 -O3 -o kmeans_openmp kmeans_openmp.c -lm -fopenmp

kmeans_pthread: kmeans_pthread.c
	gcc -std=c99 -O3 -o kmeans_pthread kmeans_pthread.c -lm -lpthread

plot: plot.txt
	gnuplot gpbatch.txt
	
plot.txt: kmeans
	./kmeans $(T) $(C) $(P) > plot.txt
	./kmeans_pthread $(T) $(C) $(P) > plott.txt

plott: 
	gnuplot gpbatcht.txt

compare: plot plott
	eog output.jpg &
	eog outputt.jpg &

jpgs: ${JPGS}

%.jpg: %.dat
	gnuplot -e "infile='$*.dat'; outfile='$*.jpg'" gpbatch.txt
	-rm -f $*.dat
	
zip:
	zip ps4.zip -o kmeans_openmp.c kmeans_pthread.c Makefile

unzip:
	unzip -o ps4.zip

upload: zip
	scp ps4.zip sondrele@clustis3.idi.ntnu.no:~/TDT4200/PS4/

run: all
	./kmeans_pthread $(T) $(C) $(P)

testmp: unzip all
	time ./kmeans $(T) $(C) $(P) > k_res
	time ./kmeans_openmp $(T) $(C) $(P) > mp_res
	diff k_res mp_res

runomp: upload
	ssh sondrele@clustis3.idi.ntnu.no 'cd TDT4200/PS4/; make testmp'

testt: unzip all
	time ./kmeans $(T) $(C) $(P) > k_res
	time ./kmeans_pthread $(T) $(C) $(P) > t_res
	diff k_res t_res

runt: upload
	ssh sondrele@clustis3.idi.ntnu.no 'cd TDT4200/PS4/; make testt'

clean:
	-rm -f kmeans plot.txt output.jpg
	-rm -f kmeans_thread plott.txt outputt.jpg
