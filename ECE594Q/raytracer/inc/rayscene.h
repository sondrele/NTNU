#ifndef _RAYSCENE_H_
#define _RAYSCENE_H_

#include <cstdlib>
#include <stdint.h>
#include <vector>

#include "Matrix.h"
#include "scene_io.h"
#include "raybuffer.h"
#include "raytracer.h"
#include "rayscene_shapes.h"

class Camera {
private:
    Vect pos;
    Vect viewDir;
    Vect orthoUp;

    float focalDist;
    float verticalFOV;

public:
    Vect getPos() const { return pos;}
    void setPos(Vect d) { pos = d;}
    Vect getViewDir() { return viewDir;}
    void setViewDir(Vect d) { viewDir = d;}
    Vect getOrthoUp() { return orthoUp;}
    void setOrthoUp(Vect d) { orthoUp = d;}
    float getFocalDist() { return focalDist;}
    void setFocalDist(float f) { focalDist = f;}
    float getVerticalFOV() { return verticalFOV;}
    void setVerticalFOV(float f) { verticalFOV = f;}
};

class Light {
private:
    Vect pos;
    Vect dir;
    PX_Color color;
    enum LightType type;

public:
    enum LightType getType() const { return type;}
    void setType(enum LightType t) { type = t;}
    PX_Color getColor() { return color;}
    void setColor(PX_Color c) { color = c;}
    Vect getPos() { return pos;}
    void setPos(Vect p) { pos = p;}
    Vect getDir() { return dir;}
    void setDir(Vect d) { dir = d;}
    
};

class RayScene {
private:
    Camera camera;
    std::vector<Light> lights;
    std::vector<Shape *> shapes;

public:
    RayScene();
    ~RayScene();
    
    void setCamera(Camera c) { camera = c;}
    Camera getCamera() { return camera;}
    void setCameraPos(Vect);

    void setLights(std::vector<Light>);
    void addLight(Light);
    Light getLight(uint);

    void setShapes(std::vector<Shape *>);
    void addShape(Shape *);
    Shape * getShape(uint);

    std::string toString() const;
    friend ostream& operator <<(ostream&, const RayScene&);
};

#endif