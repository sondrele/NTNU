#ifndef _RAYTRACER_H_
#define _RAYTRACER_H_

#include <cstdlib>
#include <cfloat>
#include <iostream>

#include "Matrix.h"
#include "rand.h"
#include "ray.h"
#include "raybuffer.h"
#include "camera.h"
#include "rayscene.h"
#include "rayscene_shapes.h"

class RayTracer {
private:
    uint WIDTH;
    uint HEIGHT;
    float M;
    uint depth;
    float scaleConst;   // c

    RayBuffer buffer;
    RayScene *scene;
    Camera camera;

    Vect parallelRight; // A: viewDir x orthogonalUp - dir of x-axis
    Vect parallelUp;    // B: A x viewDir
    Vect imageCenter;   // M: E + cV

    void calculateImagePlane();

public:
    // RayTracer(uint, uint);
    RayTracer(uint, uint, uint);
    // RayTracer(uint, uint, Vect, Vect);
    ~RayTracer();
    uint getWidth() { return WIDTH;}
    uint getHeight() { return HEIGHT;}
    void setM(float m) { M = m; }

    void setScene(RayScene *s) { scene = s; }
    void setCamera(Camera c);
    Camera getCamera() { return camera; }
    void setCameraPos(Vect);
    Vect getCameraPos() { return camera.getPos(); }

    void setViewDirection(Vect);
    Vect getViewDirection() { return camera.getViewDir(); }
    void setOrthogonalUp(Vect);
    Vect getOrthogonalUp() { return camera.getOrthoUp(); }
    Vect getParallelRight() { return parallelRight;}
    Vect getParallelUp() { return parallelUp;}
    Vect getImageCenter();

    Vect vertical();
    Vect horizontal();
    double getVerticalFOV() { return camera.getVerticalFOV(); }
    double getHorizontalFOV();
    Point_2D computePoint(uint, uint);
    Vect computeDirection(uint, uint);

    Ray computeRay(uint, uint);
    Ray computeMonteCarloRay(float, float);
    SColor calculateShadowScalar(Light &, Intersection &, int);
    SColor shadeIntersection(Intersection, int);
    RayBuffer traceRays();
    RayBuffer traceRaysWithAntiAliasing();

    // Whitted illumination
    float calculateFattj(Vect, Light *);
    SColor ambientLightning(float, SColor, SColor);
    SColor directIllumination(Light *, Intersection, SColor, float);
    SColor diffuseLightning(float, SColor, Vect, Vect);
    SColor specularLightning(float, SColor, Vect, Vect, Vect);

    // Pathtracing
    bool russianRoulette(SColor, float &);
};

#endif // _RAYTRACER_H_
