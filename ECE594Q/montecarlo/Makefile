INCLUDEPATH=\
    -I../matrix/inc\
    -I./lib\
    -I./inc
CC=clang++
CC=g++
CFLAGS = $(INCLUDEPATH) -Wall -std=c++11 -lm -fopenmp -ffast-math -g
#-pg
CIMGFLAGS = $(CFLAGS) -L/usr/X11R6/lib -lm -lpthread -lX11
LDFLAGS=
LIBS = \
    lib/scene_io.cpp\
    lib/tiny_obj_loader.cc\
    lib/rimage.cpp\
    lib/rand.cpp\
    ../matrix/src/Matrix.cpp\
    ../matrix/src/Vect_h.cpp\
    ../matrix/src/Vect.cpp
SRCS = src/argparser.cpp\
    src/progress.cpp\
    src/ray.cpp\
    src/intersection.cpp\
    src/rscene.cpp\
    src/rscenefactory.cpp\
    src/shapes.cpp\
    src/material.cpp\
    src/sphere.cpp\
    src/mesh.cpp\
    src/triangle.cpp\
    src/raybuffer.cpp\
    src/camera.cpp\
    src/shader.cpp\
    src/texture.cpp\
    src/bvhtree.cpp\
    src/envmap.cpp\
    src/directillumtracer.cpp\
    src/pathtracer.cpp\
    src/raytracer.cpp
OBJS = $(SRCS:.cpp=.o) $(LIBS:.cpp=.o) 
INCS = $(wildcard ./inc/*.h)

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

all: tracer

tracer: tracer.o $(OBJS)
	$(CC) -o $@ $^ $(CIMGFLAGS)

clean:
	rm -f *.o
	rm -rf **/*.o
