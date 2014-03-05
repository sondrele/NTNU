#ifndef _RAYSCENE_H_
#define _RAYSCENE_H_

#include <cstdlib>
#include <stdint.h>
#include <vector>

#include "Matrix.h"
#include "scene_io.h"
#include "ray.h"
#include "intersection.h"
#include "raybuffer.h"
#include "shapes.h"
#include "bvhtree.h"

class Light {
private:
    Vect pos;
    Vect dir;
    enum LightType type;
    SColor intensity;
    
public:
    enum LightType getType() const { return type;}
    void setType(enum LightType t) { type = t;}
    Vect getPos() { return pos;}
    void setPos(Vect p) { pos = p;}
    Vect getDir() { return dir;}
    void setDir(Vect d) { dir = d;}
    SColor getIntensity() { return intensity; }
    void setIntensity(SColor i) { intensity = i; }
};

class RScene {
private:
    std::vector<Light *> lights;
    std::vector<Shape *> shapes;
    BVHTree searchTree;

public:
    RScene();
    ~RScene();

    void setLights(std::vector<Light *>);
    void addLight(Light *);
    Light * getLight(uint);
    std::vector<Light *> getLights() { return lights; }

    void setShapes(std::vector<Shape *>);
    void addShape(Shape *);
    Shape * getShape(uint);
    Intersection calculateRayIntersection(Ray);
    Intersection intersectsWithBVHTree(Ray);
};

#endif