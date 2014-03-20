#ifndef _RAYTRACER_H_
#define _RAYTRACER_H_

#include <cstdlib>
#include <cfloat>
#include <iostream>
#include <assert.h>

#ifndef _WINDOWS
#include "omp.h"
#endif  // _WINDOWS


#include "progress.h"
#include "envmap.h"
#include "Matrix.h"
#include "rand.h"
#include "ray.h"
#include "raybuffer.h"
#include "camera.h"
#include "rscene.h"
#include "shapes.h"

class RayTracer {
protected:
    uint WIDTH;
    uint HEIGHT;
    int numSamples;
    uint depth;
    float scaleConst;   // c
    float fattjScale;

    bool usingEnvMap;

    RayBuffer buffer;
    RScene *scene;
    Camera camera;
    EnvMap envMap;

    Vect parallelRight; // A: viewDir x orthogonalUp - dir of x-axis
    Vect parallelUp;    // B: A x viewDir
    Vect imageCenter;   // M: E + cV

    void calculateImagePlane();

public:
    RayTracer();
    RayTracer(uint, uint, uint);
    virtual ~RayTracer();
    // Setters and getters
    uint getWidth() { return WIDTH; }
    uint getHeight() { return HEIGHT; }
    void setFattjScale(float fs) { fattjScale = fs; }
    void setNumSamples(int s) { numSamples = s; }
    void setScene(RScene *s) { scene = s; }
    void setCamera(Camera c);
    Camera getCamera() { return camera; }
    void setCameraPos(Vect);
    Vect getCameraPos() { return camera.getPos(); }
    void setViewDirection(Vect);
    Vect getViewDirection() { return camera.getViewDir(); }
    void setOrthogonalUp(Vect);
    Vect getOrthogonalUp() { return camera.getOrthoUp(); }
    Vect getParallelRight() { return parallelRight; }
    Vect getParallelUp() { return parallelUp; }
    Vect getImageCenter();
    // Util functions for setting up scenes
    void loadEnvMap(std::string);
    Vect vertical();
    Vect horizontal();
    double getVerticalFOV() { return camera.getVerticalFOV(); }
    double getHorizontalFOV();
    Point_2D computePoint(uint, uint);
    Vect computeDirection(uint, uint);
    // Ray tracing functions
    Ray computeRay(float, float);
    SColor calculateShadowScalar(Light *, Intersection &, int, int);

    // Direct illumination
    SColor diffuseLightning(float, SColor, Vect, Vect);
    SColor specularLightning(float, SColor, Vect, Vect, Vect);
    SColor directIllumination(Light *, Intersection, SColor, float);
    float calculateFattj(Vect, Light *);

    virtual RayBuffer traceScene() = 0;
};

class WhittedTracer : public RayTracer {
public:
    WhittedTracer(uint, uint, uint);
    virtual ~WhittedTracer();

    // Whitted illumination
    SColor ambientLightning(float, SColor, SColor);
    SColor shadeIntersection(Intersection, int);

    virtual RayBuffer traceScene();
};

class PathTracer : public RayTracer {
public:
    PathTracer(uint, uint, uint);
    virtual ~PathTracer();
    
    // Pathtracing
    bool russianRoulette(SColor, float &);
    SColor diffuseInterreflect(Intersection , int);
    Vect uniformSampleUpperHemisphere(Vect &);
    Vect specularSampleUpperHemisphere(Intersection &ins);
    SColor shadeIntersectionPath(Intersection in, int d);

    virtual RayBuffer traceScene();
};

class BiPathTracer : public RayTracer {
private:
    bool bidirectional;

public:
    BiPathTracer(uint, uint, uint);
    virtual ~BiPathTracer();

    void setBidirectional(bool b) { bidirectional = b; }
    Ray computeaRayFromLightSource(Light *);
    Light * pickRandomLight();
    Vect specularSampleUpperHemisphere(Intersection &);
    SColor connectPaths(Vect &, Vect &, SColor &);
    bool russianRoulette(SColor);
    float fattj(Vect, Vect);
    SColor shootRayFromLightSource(Light *, Vect &, int);
    bool traceRayFromCamera(uint, uint, SColor &, int, int);
    SColor shadeIntersectionPoint(Intersection &, Vect &, int &, int, bool);

    // Bidirectional path tracing
    virtual RayBuffer traceScene();
};

#endif // _RAYTRACER_H_
