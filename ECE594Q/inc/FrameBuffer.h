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
    float *samples;
    std::vector<Vect> points;
    std::vector<Mesh *> objects;
    void initSamples();

public:
    FrameBuffer(uint, uint);
    FrameBuffer(uint, uint, float, float, float, float);
    FrameBuffer(uint, uint, uint, uint);
    ~FrameBuffer();
    
    float *getSamples() { return samples; };
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