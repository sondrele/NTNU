INCLUDEPATH=\
    -I../matrix/inc\
    -Ilib\
    -I./inc
CC=clang++
CC=g++
CFLAGS = $(INCLUDEPATH) -Wall -std=c++11 -lm -O3 -fopenmp -ffast-math -g
#-pg
CIMGFLAGS = $(CFLAGS) -L/usr/X11R6/lib -lm -lpthread -lX11
LDFLAGS=
LIBS = \
    lib/scene_io.cpp\
    lib/rayimage.cpp\
    ../matrix/src/Matrix.cpp\
    ../matrix/src/Vect.cpp
SRCS = src/ray.cpp\
    src/intersection.cpp\
    src/camera.cpp\
    src/raybuffer.cpp\
    src/rayscene.cpp\
    src/rayscene_factory.cpp\
    src/rayscene_shapes.cpp\
    src/sphere.cpp\
    src/mesh.cpp\
    src/triangle.cpp\
    src/material.cpp\
    src/shader.cpp\
    src/texture.cpp\
    src/bvhtree.cpp\
    src/rand.cpp\
    src/progress.cpp\
    src/envmap.cpp\
    src/raytracer.cpp
OBJS = $(SRCS:.cpp=.o) $(LIBS:.cpp=.o) 
INCS = $(wildcard ./inc/*.h)

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

all: montecarlo

montecarlo: montecarlo.o $(OBJS)
	$(CC) -o $@ $^ $(CIMGFLAGS)

cornell: montecarlo
	time ./montecarlo cornell 300 300 5 10

clean:
	rm -f *.o
	rm -rf **/*.o

