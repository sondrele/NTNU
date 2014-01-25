#ifndef _MESH_H_
#define _MESH_H_

#include <cmath>
#include <vector>
#include <iostream>
#include <sstream>
#include <cstdint>

#include "Matrix.h"
#include "Utils.h"

class MeshPoint : public Vect {
public:
    float textureCoordinate[2];
    unsigned char color[3];

    MeshPoint(Vect &v);
    MeshPoint(float, float, float);
    MeshPoint() : MeshPoint(0, 0, 0) {;};
};

typedef struct {
    int X_start;
    int Y_start;
    int X_stop;
    int Y_stop;
} BoundingBox;

class MicroPolygon {
private:
    unsigned char color[3];
    // Vect surfaceNormal;
    
public:
    MeshPoint a;
    MeshPoint b;
    MeshPoint c;
    MeshPoint d;

    void setColor(unsigned char *);
    void setColor(unsigned char, unsigned char, unsigned char);
    unsigned char *getColor() { return color; };
    bool isNeighbour(MicroPolygon other);
    bool intersects(Vect point);
    BoundingBox getBoundingBox();
    float *getBound();
};

class Mesh {
protected:
    uint width;
    uint height;
    std::vector<MeshPoint> points;

public:
    Mesh(uint, uint);
    ~Mesh();
    
    uint getSize();
    uint getWidth() const { return width; };
    uint getHeight() const { return height; };

    void addPoint(MeshPoint);
    MeshPoint getPoint(uint);
    std::string toString();
    std::vector<MicroPolygon> getMicroPolygons();
    
    void rotate(const char, const float);
    void translate(const float, const float, const float);
};

class RiSphere : public Mesh {
private:
    float radius;
    unsigned int resolution;

public:
    RiSphere(float radius, uint);
    float getRadius() const { return radius; };
    void setRadius(float radius);
    std::string toString();
};

class RiRectangle : public Mesh {
private:
    float width;
    float height;
    float depth;

public:
    RiRectangle(float, float, float);
};

#endif // _MESH_H_