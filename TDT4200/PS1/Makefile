
INCLUDEPATH=\
    -Iinclude\
    -I/usr/local/include
CC ?= cc
CFLAGS += -std=c99 ${INCLUDEPATH} -g
LDFLAGS += -L/usr/local/lib
SRCS = src/linkedlist.c
OBJS = ${SRCS:.c=.o}
TEST = lib/ctest.c test/test_linkedlist.c
TOBJ = ${TEST:.c=.o} ${OBJS}
DEPS = linkedlist.h

%.o: %.c ${DEPS}
	${CC} ${CFLAGS} -c -o $@ $<

# Default
all: setup linkedlist

setup: bin obj
	@echo "SRCS = ${SRCS}"
	@echo "TEST = ${TEST}"
	@echo "OBJS = ${OBJS}"

linkedlist: ${OBJS}
	${CC} -o bin/$@ $^ ${CFLAGS}
	mv src/*.o obj/
#mv test/*.o obj/

test: all ${TOBJ}
	@strip -N main obj/linkedlist.o -o obj/_linkedlist.o
	${CC} -o bin/$@ ${TEST:.c=.o} obj/_linkedlist.o ${CFLAGS}
	bin/test

run: all
	bin/linkedlist

# Cleanup.
.PHONY: clean

clean:
	if [ -e obj ]; then rm -r obj; fi

purge: clean
	if [ -e bin ]; then rm -r bin; fi

# Targets to create directories (when they don't exist already).
bin:
	mkdir bin
obj:
	mkdir obj