#ifndef _FRAMEBUFFER_H_
#define _FRAMEBUFFER_H_

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

#include "CImg.h"
#include "Matrix.h"
#include "Mesh.h"

typedef struct {
    unsigned char R;
    unsigned char G;
    unsigned char B;
} F_Color;

class FramePixel {
private:
    uint X;
    uint Y;
    std::vector<F_Color> samples;

public:
    FramePixel(uint X, uint Y);
    void addSample(F_Color);
    void addSample(unsigned char*);
    F_Color getColor();
    uint getX() { return X; };
    uint getY() { return Y; };
};

class FrameBuffer {
private:
    uint WIDTH;
    uint HEIGHT;
    float hither;
    float yon;
    float fov;
    float aspectRatio;
    uint m;
    uint n;
    std::vector<Vect> points;
    std::vector<Mesh *> objects;
    std::vector<FramePixel> pixels;

public:
    FrameBuffer(uint, uint);
    FrameBuffer(uint, uint, float, float, float, float);
    FrameBuffer(uint, uint, uint, uint);

    void addPoint(Vect);
    Matrix getPoint(unsigned int);
    void addMesh(Mesh &);
    void plotPoints(const char *);
    void drawMicroPolygons(const char *);
    void projectAndScalePoint(Vect &);
    void projectMeshPoint(MeshPoint &);
    void projectMicroPolygon(MicroPolygon &);
};

#endif // _FRAMEBUFFER_H_