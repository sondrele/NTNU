#ifndef _MESH_H_
#define _MESH_H_

#include <cmath>
#include <vector>
#include <iostream>
#include <sstream>
#include <cstdint>

#include "Matrix.h"
#include "Utils.h"

class MeshPoint {
public:
    MeshPoint() : MeshPoint(0, 0, 0) {;};
    MeshPoint(float, float, float);
    
    Vect point;
    Vect surfaceNormal;
    float textureCoordinate[2];
    unsigned char color[3];

    float getX() { return point.getX(); };
    void setX(float x) { point.setX(x); };
    float getY() { return point.getY(); };
    void setY(float y) { point.setY(y); };
    float getZ() { return point.getZ(); };
    void setZ(float z) { point.setZ(z); };
    float getW() { return point.getW(); };
    void setW(float w) { point.setW(w); };
};

class MicroPolygon {
private:
    unsigned char color[3];
    
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
    float* getBoundingBox();
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
    void addPoint(MeshPoint);
    MeshPoint getPoint(uint);
    void movePoint(uint, float, float, float);
    std::string toString();
    std::vector<MicroPolygon> getMicroPolygons();
    int getWidth() const { return width; };
    int getHeight() const { return height; };
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
    void rotate(float, char);
    void translate(float, float, float);
};

class RiRectangle : public Mesh {
private:
    float width;
    float height;
    float deapth;

public:
    RiRectangle(float, float, float);
};

#endif // _MESH_H_