CC ?= cc
MPICC ?= mpicc
INCS =
LIBS =
CFLAGS += -g -std=c99 -pedantic -Wall ${INCS}
LDFLAGS += -static ${LIBS}
SRC = linkedlist.c
SRC2 = ring.c
OBJ = ${SRC:.c=.o}
OBJ2 = ${SRC2:.c=.o}

all: options linkedlist test_runner ring test

options:
	@echo linkedlist build options:
	@echo "CFLAGS   = ${CFLAGS}"
	@echo "LDFLAGS  = ${LDFLAGS}"
	@echo "CC       = ${CC}"
	@echo "MPICC    = ${MPICC}"

.c.o:
	@echo CC $<
	@${CC} -c ${CFLAGS} $<

linkedlist: ${OBJ}
	@echo CC -o $@
	@${CC} -o $@ ${OBJ} ${LDFLAGS}

test_runner: test_runner.o linkedlist.o
	@echo CC -o $@
	@echo Stubbing symbols: main
	@strip -N main linkedlist.o -o _linkedlist.o
	@${CC} -o $@ test_runner.o _linkedlist.o ${LDFLAGS}

ring: ${SRC2}
	@echo ${MPICC} ${CFLAGS} -o $@ ${SRC2}
	@${MPICC} ${CFLAGS} -o $@ ${SRC2}

test: test_runner
	@echo Running tests...
	@./test_runner

run: linkedlist.o
	./linkedlist

clean:
	@echo cleaning
	@rm -f test_runner *.o *.core linkedlist