#ifndef _FRAMEBUFFER_H_
#define _FRAMEBUFFER_H_

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cfloat>

#include <vector>
#include <list>
#include <algorithm>

#include "CImg.h"
#include "Matrix.h"
#include "Mesh.h"

typedef struct {
    unsigned char R;
    unsigned char G;
    unsigned char B;
} PX_Color;

typedef struct {
    float depth;
    float opacity;
    PX_Color color;
} PX_Sample;

class Sample {
private:
    std::list<PX_Sample> samples;

public:
    Sample();
    void addSample(PX_Sample);
    PX_Sample getSample();
    PX_Color getColor();
    std::string toString();
};

class FramePixel {
private:
    uint x;
    uint y;
    uint m;
    uint n;
    std::vector<Sample> samplePoints;

public:
    FramePixel(uint X, uint Y, uint m, uint n);
    // void setSample(uint x, uint y, PX_Color);
    // void setSample(uint x, uint y, unsigned char *);
    void setSample(uint x, uint y, PX_Sample s);
    PX_Color getColor();
    uint getX() { return x; };
    uint getM() { return m; };
    uint getY() { return y; };
    uint getN() { return n; };
};

class FrameBuffer {
private:
    uint WIDTH;
    uint HEIGHT;
    uint m;
    uint n;
    float hither;
    float yon;
    float fov;
    float aspectRatio;
    std::vector<FramePixel> pixels;
    std::vector<MeshPoint> points;
    std::vector<MicroPolygon> mPolygons;
    std::vector<Mesh *> objects;

public:
    FrameBuffer(uint, uint, uint, uint, float, float, float, float);
    FrameBuffer(uint, uint);
    FrameBuffer(uint, uint, uint, uint);

    void setPixel(uint, uint, uint, uint, PX_Sample); 
    FramePixel getPixel(uint x, uint y);
    uint getSize() { return pixels.size(); };

    void addPoint(MeshPoint);
    MeshPoint getPoint(uint);
    void addMesh(Mesh &);
    void addMicroPolygon(MicroPolygon &);
    void plotPoints(const char *);
    void drawShapes(const char *);
    void exportImage(const char *);
    bool projectAndScalePoint(Vect &);
    bool projectMicroPolygon(MicroPolygon &);
};

#endif // _FRAMEBUFFER_H_