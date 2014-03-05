#ifndef _SHAPES_H_
#define _SHAPES_H_

#include <cstdlib>
#include <stdint.h>
#include <vector>
#include <memory>
#include <sstream>

#include "Matrix.h"
#include "scene_io.h"
#include "ray.h"
#include "material.h"
#include "texture.h"
#include "shader.h"
#include "intersection.h"
#include "raybuffer.h"

enum ShapeType {
    SPHERE, MESH, TRIANGLE
};

class BBox {
protected:
    Vect pmin;
    Vect pmax;

public:
    BBox();
    void setMin(Vect ll) { pmin = ll; }
    Vect getMin() const { return pmin; }
    void setMax(Vect ur) { pmax = ur; }
    Vect getMax() const { return pmax; }
    bool intersects(Ray);
    Vect getCentroid() const;

    const friend BBox operator +(const BBox &, const BBox &);
};

bool operator < (const BBox&, const BBox&);

class Intersection;

class Shape {
protected:
    std::vector<Material *> materials;
    Texture *texture;
    CShader *cShader;
    IShader *iShader;

public:
    Shape();
    virtual ~Shape();
    
    virtual ShapeType getType() = 0;
    virtual Intersection intersects(Ray) = 0;
    virtual Vect surfaceNormal(Vect, Vect) = 0;
    virtual Material * getMaterial() = 0;
    virtual SColor getColor(Vect) = 0;
    virtual BBox getBBox() = 0;
    uint64_t getNumMaterials() { return materials.size(); }
    void addMaterial(Material *m);

    void setTexture(Texture *);
    Texture * getTexture();
    bool hasTexture() { return texture != NULL; }

    void setCShader(CShader *);
    CShader * getCShader();
    void setIShader(IShader *s);
    IShader * getIShader();

    static bool CompareX(Shape *, Shape *);
    static bool CompareZ(Shape *, Shape *);
    static bool CompareY(Shape *, Shape *);
};

bool operator < (Shape&, Shape&);

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
    virtual ~Sphere();
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

    Point_2D getUV(Vect);

    virtual Material * getMaterial();
    virtual Intersection intersects(Ray);
    virtual Vect surfaceNormal(Vect, Vect);
    virtual SColor getColor(Vect);
    virtual BBox getBBox();
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
    Material * getMaterial(char);

    virtual Material * getMaterial();
    virtual Intersection intersects(Ray);
    virtual Vect surfaceNormal(Vect, Vect);
    virtual SColor getColor(Vect);
    virtual BBox getBBox();

    Vect normal();
    Vect interpolatedNormal(Vect);

    float getArea();
    static float getArea(Vect, Vect, Vect);
};

class Mesh : public Shape {
private:
    std::vector<Triangle *> triangles;
    bool textureCoords;
    bool hasPerSurfaceNormal;
    bool objectMaterial;
    BBox bbox;

public:
    Mesh();
    virtual ~Mesh();
    virtual ShapeType getType() { return MESH; }

    void addTriangle(Triangle *t);
    Triangle * getTriangle(uint64_t i) { return triangles.at(i);}
    std::vector<Triangle *> getTriangles() { return triangles; }
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
    virtual SColor getColor(Vect);
    virtual BBox getBBox();
};

#endif // _SHAPES_H_
