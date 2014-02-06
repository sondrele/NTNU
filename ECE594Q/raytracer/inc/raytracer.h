#ifndef _RAYTRACER_H_
#define _RAYTRACER_H_

#include <cstdlib>
#include <iostream>
#include "Matrix.h"

typedef struct {
    float x;
    float y;
} Point_2D;

class Ray {
private:
    Vect origin;
    Vect direction;

public:
    Ray();
    Ray(Vect, Vect);
    Vect getOrigin() { return origin;}
    void setOrigin(Vect o) { origin = o;}
    Vect getDirection() { return direction;}
    void setDirection(Vect d) { direction = d;}

};

class RayTracer {
private:
    uint WIDTH;
    uint HEIGHT;
    double verticalFOV; // vertical field of view
    float scaleConst;

    Vect cameraPos; // E
    Vect viewDirection; // V
    Vect orthogonalUp; // U

    Vect parallelRight; // A: viewDir x orthogonalUp - dir of x-axis
    Vect parallelUp; // B: A x viewDir
    Vect imageCenter; // M: E + cV

    void calculateImagePlane();

public:
    RayTracer(uint, uint);
    RayTracer(uint, uint, Vect, Vect);

    uint getWidth() { return WIDTH;}
    uint getHeight() { return HEIGHT;}
    void setCamera(Vect cam);
    Vect getCamera() { return cameraPos;}
    void setViewDirection(Vect);
    Vect getViewDirection() { return viewDirection;}
    void setOrthogonalUp(Vect);
    Vect getOrthogonalUp() { return orthogonalUp;}
    Vect getParallelRight() { return parallelRight;}
    Vect getParallelUp() { return parallelUp;}
    Vect getImageCenter() { return imageCenter;}

    Vect vertical();
    Vect horizontal();
    double getVerticalFOV() { return verticalFOV;}
    double getHorizontalFOV();
    Point_2D computePoint(uint, uint);
    Vect computeDirection(uint, uint);
    Ray computeRay(uint, uint);
};

#endif // _RAYTRACER_H_
