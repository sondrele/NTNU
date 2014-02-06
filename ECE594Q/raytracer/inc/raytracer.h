#ifndef _RAYTRACER_H_
#define _RAYTRACER_H_

#include <cstdlib>
#include <iostream>

#include "Matrix.h"
#include "ray.h"
#include "raybuffer.h"
#include "rayscene.h"

typedef struct {
    float x;
    float y;
} Point_2D;

class RayTracer {
private:
    uint WIDTH;
    uint HEIGHT;
    float scaleConst;   // c

    Vect parallelRight; // A: viewDir x orthogonalUp - dir of x-axis
    Vect parallelUp;    // B: A x viewDir
    Vect imageCenter;   // M: E + cV

    RayBuffer buffer;
    RayScene scene;

    void calculateImagePlane();

public:
    RayTracer(uint, uint);
    RayTracer(uint, uint, Vect, Vect);

    uint getWidth() { return WIDTH;}
    uint getHeight() { return HEIGHT;}
    void setCameraPos(Vect cam);
    Camera getCamera() { return scene.getCamera();}
    Vect getCameraPos() { return getCamera().getPos(); }
    void setViewDirection(Vect);
    Vect getViewDirection() { return scene.getCamera().getViewDir(); }
    void setOrthogonalUp(Vect);
    Vect getOrthogonalUp() { return scene.getCamera().getOrthoUp(); }
    Vect getParallelRight() { return parallelRight;}
    Vect getParallelUp() { return parallelUp;}
    Vect getImageCenter();

    Vect vertical();
    Vect horizontal();
    double getVerticalFOV() { return scene.getCamera().getVerticalFOV(); }
    double getHorizontalFOV();
    Point_2D computePoint(uint, uint);
    Vect computeDirection(uint, uint);
    Ray computeRay(uint, uint);
    RayBuffer traceRays();

    RayBuffer getRayBuffer() { return buffer; }
};

#endif // _RAYTRACER_H_
