
INCLUDE = -I../PS1/include/
CC = mpicc
REALCC = cc
CFLAGS = -std=c99 -g $(INCLUDE)

CTEST = ../PS1/lib/ctest.c jacobi.c test_jacobi.c
MPITEST = mpi_test.c fluid2d.c bmp.c jacobi.c
TEST = mpi_test
SRC = fluid2d.c main.c bmp.c jacobi.c
BIN = fluid2d
ZIP = ps2.zip

SSH = sondrele@clustis3.idi.ntnu.no

# Fluid
fluid: $(BIN)

$(BIN) : $(SRC)

run: unzip $(BIN)
	qrsh -cwd mpirun -n 8 $(BIN) 1 8

runfluid: upload
	ssh $(SSH) 'cd /home/sondrele/TDT4200/PS2/;make run'

# Mpitest
mpitest: $(TEST)

$(TEST) : $(MPITEST)

runtest: unzip $(TEST)
	qrsh -cwd mpirun -n 8 $(TEST)

runmpitest: upload
	ssh $(SSH) 'cd /home/sondrele/TDT4200/PS2/;make runtest'

# Server communication
upload:
	zip -r $(ZIP) .
	scp $(ZIP) $(SSH):~/TDT4200/PS2/

unzip:
	unzip -o $(ZIP)

download:
	scp sondrele@clustis3.idi.ntnu.no:~/TDT4200/PS2/out.bmp .
	eog ./out.bmp

runtests: $(CTEST)
	$(REALCC) -o $@ $(CTEST) $(CFLAGS)
	./$@

clean:
	rm -f $(BIN) *.o
