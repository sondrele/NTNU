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
#include "intersection.h"
#include "raybuffer.h"

typedef struct {
    float x;
    float y;
} Point_2D;

class SColor : public Vect {
public:
    SColor() {}
    SColor(Vect);
    SColor(Color);
    SColor(float, float, float);
    float R() { return getX(); }
    void R(float);
    float G() { return getY(); }
    void G(float);
    float B() { return getZ(); }
    void B(float);

    SColor& operator=(const Vect&);
};

class Material {
private:
    SColor diffColor;
    SColor ambColor;
    SColor specColor;
    // SColor emissColor;
    float shininess;
    float transparency;

public:
    Material();
    // void setEmissColor(SColor c) { emissColor = c; }
    // SColor getEmissColor() { return emissColor; }
    void setDiffColor(SColor c) { diffColor = c; }
    SColor getDiffColor() { return diffColor; }
    void setAmbColor(SColor c) { ambColor = c; }
    SColor getAmbColor() { return ambColor; }
    void setSpecColor(SColor c) { specColor = c; }
    SColor getSpecColor() { return specColor; }
    void setShininess(float c) { shininess = c; }
    float getShininess() { return shininess; }
    void setTransparency(float c) { transparency = c; }
    float getTransparency() { return transparency; }

    bool hasTexture();
    bool isReflective();
    bool isRefractive();
};

enum ShapeType {
    SPHERE, MESH, TRIANGLE
};

class Intersection;

class Shape {
protected:
    std::vector<Material *> materials;

public:
    Shape();
    virtual ~Shape();
    
    virtual ShapeType getType() = 0;
    virtual Intersection intersects(Ray) = 0;
    virtual Vect surfaceNormal(Vect, Vect) = 0;
    virtual Material * getMaterial() = 0;

    uint64_t getNumMaterials() { return materials.size(); }
    void addMaterial(Material *m) { materials.push_back(m); }
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
    virtual ShapeType getType() { return SPHERE; }

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

    Point_2D getLongAndLat(Vect);

    virtual Material * getMaterial();
    virtual Intersection intersects(Ray);
    virtual Vect surfaceNormal(Vect, Vect);
};

class Plane {
private:
    Vect normal;
    Vect point;

public:
    Plane() {}
    Plane(Vect n, Vect p) : normal(n), point(p) {}
    Intersection intersects(Ray);
};

class Vertex : public Vect {
private:
    uint matIndex;
    Point_2D coords;
    bool hasNormal;
    Vect normal;
    Material *mat;

public:
    Vertex();
    Vertex(float, float, float);
    void setMaterialIndex(uint i) { matIndex = i; }
    uint getMaterialIndex() { return matIndex; }
    void setMaterial(Material *m) { mat = m; }
    Material * getMaterial() { return mat; } 
    void setSurfaceNormal(Vect n);
    Vect getSurfaceNormal();
    void setTextureCoords(float, float);
    Point_2D getTextureCoords();

    Vertex& operator=(const Vect&);
};

class Triangle : public Shape {
private:
    Vertex a;
    Vertex b;
    Vertex c;
    bool hasPerVertexNormal;
    bool hasPerVertexMaterial;

public:
    Triangle();
    virtual ~Triangle();
    virtual ShapeType getType() { return TRIANGLE; }

    Vertex getA() { return a;}
    void setA(Vertex x) { a = x;}
    Vertex getB() { return b;}
    void setB(Vertex y) { b = y;}
    Vertex getC() { return c;}
    void setC(Vertex z) { c = z;}
    void setMaterial(Material *m, char);
    void setPerVertexNormal(bool);
    void setPerVertexMaterial(bool);

    virtual Material * getMaterial();
    Material * getMaterial(char);
    virtual Intersection intersects(Ray);
    virtual Vect surfaceNormal(Vect, Vect);
    Vect normal();
    Vect interpolatedNormal(Vect);
    SColor interpolatedColor(Vect);

    float getArea();
    static float getArea(Vect, Vect, Vect);
};

class Mesh : public Shape {
private:
    std::vector<Triangle *> triangles;
    bool textureCoords;
    bool hasPerSurfaceNormal;
    bool objectMaterial;

public:
    Mesh();
    virtual ~Mesh();
    virtual ShapeType getType() { return MESH; }

    void addTriangle(Triangle *t);
    Triangle * getTriangle(uint64_t i) { return triangles.at(i);}
    uint64_t size() { return triangles.size();}
    void hasTextureCoords(bool coords) { textureCoords = coords; }
    bool hasTextureCoords() { return textureCoords; }
    void perVertexMaterial(bool mat) { objectMaterial = mat; }
    bool perVertexMaterial() { return objectMaterial; }
    void perVertexNormal(bool norm) { hasPerSurfaceNormal = norm; }
    bool perVertexNormal() { return hasPerSurfaceNormal; }

    virtual Material * getMaterial();
    virtual Intersection intersects(Ray);
    virtual Vect surfaceNormal(Vect, Vect);
};

#endif // _RAYSCENE_SHAPES_H_
