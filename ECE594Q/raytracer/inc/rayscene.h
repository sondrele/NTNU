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
#include "rayscene_shapes.h"

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
    void setIntensity(SColor);
    SColor getIntensity();
};

class RayScene {
private:
    std::vector<Light *> lights;
    std::vector<Shape *> shapes;

public:
    RayScene();
    ~RayScene();

    void setLights(std::vector<Light *>);
    void addLight(Light *);
    Light * getLight(uint);
    std::vector<Light *> getLights() { return lights; }

    void setShapes(std::vector<Shape *>);
    void addShape(Shape *);
    Shape * getShape(uint);
    Intersection calculateRayIntersection(Ray);

    std::string toString() const;
    friend ostream& operator <<(ostream&, const RayScene&);
};

#endif