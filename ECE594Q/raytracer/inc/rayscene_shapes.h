#ifndef _RAYSCENE_SHAPES_H_
#define _RAYSCENE_SHAPES_H_

#include <cstdlib>
#include <stdint.h>
#include <vector>
#include <memory>
#include <sstream>

#include "Matrix.h"
#include "scene_io.h"
#include "ray.h"
#include "raybuffer.h"

class Material {
private:
    Vect diffColor;
    Vect ambColor;
    Vect specColor;
    float shininess;
    float transparency;

public:
    Material();
    void setDiffColor(Vect c) { diffColor = c; }
    Vect getDiffColor() { return diffColor; }
    void setAmbColor(Vect c) { ambColor = c; }
    Vect getAmbColor() { return ambColor; }
    void setSpecColor(Vect c) { specColor = c; }
    Vect getSpecColor() { return specColor; }
    void setShininess(float c) { shininess = c; }
    float getShininess() { return shininess; }
    void setTransparency(float c) { transparency = c; }
    float getTransparency() { return transparency; }
};

enum ShapeType {
    SPHERE, MESH, TRIANGLE
};

class Shape {
private:
    std::vector<Material> materials;
public:
    virtual ~Shape() {}
    
    virtual bool intersects(Ray, float &) = 0;
    virtual ShapeType getType() = 0;

    Material getMaterial(uint i) { return materials.at(i); }
    void setMaterial(Material m) { materials.push_back(m); }
};


class Sphere : public Shape {
private:
    Vect origin;
    Vect xaxis;
    Vect yaxis;
    Vect zaxis;

    float radius;
    float xlength;
    float ylength;
    float zlength;

public:
    Sphere();
    virtual ~Sphere() {}
    virtual ShapeType getType();

    Vect getOrigin() { return origin;}
    void setOrigin(Vect o) { origin = o;}
    float getRadius() { return radius;}
    void setRadius(float r) { radius = r;}
    Vect getX() { return xaxis;}
    float getXlen() { return xlength;}
    void setX(float xlen, Vect x) { xlength = xlen; xaxis = x;}
    Vect getY() { return yaxis;}
    float getYlen() { return ylength;}
    void setY(float ylen, Vect y) { ylength = ylen; yaxis = y;}
    Vect getZ() { return zaxis;}
    float getZlen() { return zlength;}
    void setZ(float zlen, Vect z) { zlength = zlen; zaxis = z;}

    virtual bool intersects(Ray, float &);
};

class Triangle : public Shape {
private:
    Vect a;
    Vect b;
    Vect c;

public:
    virtual ~Triangle() {}
    virtual ShapeType getType();

    Vect getA() { return a;}
    void setA(Vect x) { a = x;}
    Vect getB() { return b;}
    void setB(Vect y) { b = y;}
    Vect getC() { return c;}
    void setC(Vect z) { c = z;}
    virtual bool intersects(Ray, float &);

};

class Mesh : public Shape {
private:
    std::string name;
    std::vector<Triangle> triangles;

public:
    virtual ~Mesh() {}
    virtual ShapeType getType();

    void addTriangle(Triangle t);
    Triangle getTriangle(uint64_t i) { return triangles.at(i);}
    uint64_t size() { return triangles.size();}

    virtual bool intersects(Ray, float &);
};

#endif // _RAYSCENE_SHAPES_H_
