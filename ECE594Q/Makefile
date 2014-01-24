INCLUDEPATH=\
    -Iinc
CC=clang++
CC=g++
CFLAGS = $(INCLUDEPATH) -Wall -g -std=c++11
CIMGFLAGS = $(CFLAGS) -O2 -L/usr/X11R6/lib -lm -lpthread -lX11
LDFLAGS=
SRCS = lib/Utils.cpp lib/Matrix.cpp lib/Mesh.cpp lib/FrameBuffer.cpp
OBJS = $(SRCS:.cpp=.o)
TEST=lib/ctest.c

DEPS = Mesh.h
%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

all: matrix_test fb_test mesh_test main

test: all
	./matrix_test
	./fb_test
	./mesh_test

MAT = $(TEST) lib/Matrix.cpp test/test_matrix.cpp
MATOBJS = $(MAT:.cpp=.o)
matrix_test: $(MATOBJS)
	$(CC) -o $@ $^ $(CFLAGS)

FB = $(TEST) lib/Utils.cpp lib/Matrix.cpp lib/Mesh.cpp lib/FrameBuffer.cpp test/test_framebuffer.cpp
FBOBJS = $(FB:.cpp=.o)
fb_test: $(FBOBJS)
	$(CC) -o $@ $^ $(CIMGFLAGS)

MESH = $(TEST) lib/Utils.cpp lib/Matrix.cpp lib/Mesh.cpp test/test_mesh.cpp
MESHOBJS = $(MESH:.cpp=.o)
mesh_test: $(MESHOBJS)
	$(CC) -o $@ $^ $(CFLAGS)

main: $(OBJS) main.o
	$(CC) -o $@ $^ $(CIMGFLAGS)

clean:
	rm -rf **/*.o