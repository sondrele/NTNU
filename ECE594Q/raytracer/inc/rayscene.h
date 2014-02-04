#ifndef _RAYSCENE_H_
#define _RAYSCENE_H_

#include <vector>

#include "Matrix.h"
#include "scene_io.h"

class Camera {
private:
    Vect pos;
    Vect viewDir;
    Vect orthoUp;
    double focalDist;
    double verticalFOV;

public:
    Camera();

};

class Polygon {
private:

public:
};

class Sphere {
private:

public:
};

class Triangle {
private:
    std::string name;

public:
};

typedef struct {
    Vect pos;
    PX_Color color;
} Light;


class RayScene {
private:
    Camera cam;
    std::vector<Polygon> shapes;
    std::vector<Light> lights;

public:
    RayScene(SceneIO *);
};

#endif