
INCLUDE = -I../PS1/include/
CC = cc
MPICC = mpicc
CFLAGS = -std=c99 -g $(INCLUDE)

CTEST = ../PS1/lib/ctest.c jacobi.c test_jacobi.c
SRC = fluid2d.c main.c bmp.c jacobi.c
BIN = fluid2d
ZIP = ps2.zip

fluid: $(BIN)

run: unzip $(BIN)
	qrsh -cwd mpirun -n 8 $(BIN) 50 256
	
unzip:
	unzip -o $(ZIP)

$(BIN) : $(SRC)

upload:
	zip $(ZIP) $(SRC)
	scp $(ZIP) sondrele@clustis3.idi.ntnu.no:~/TDT4200/PS2/

test: $(CTEST)
	$(CC) -o $@ $(CTEST) $(CFLAGS)
	./$@

clean:
	rm -f $(BIN) *.o















