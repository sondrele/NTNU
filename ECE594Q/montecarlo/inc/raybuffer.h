#ifndef _RAYBUFFER_H_
#define _RAYBUFFER_H_

#include <cstdint>
#include "uint.h"
#include <stdlib.h>
#include <vector>

#define PX(x, y) (y * WIDTH + x)

typedef struct {
    uint8_t R;
    uint8_t G;
    uint8_t B;
} PX_Color;

class RayPixel {
private:
    uint X;
    uint Y;
    float opacity;
    PX_Color color;

public:
    RayPixel(uint, uint);
    uint getX() { return X; }
    uint getY() { return Y; }
    void setColor(PX_Color);
    void setColor(uint8_t, uint8_t, uint8_t);
    PX_Color getColor() { return color; }
};

class RayBuffer {
private:
    uint WIDTH;
    uint HEIGHT;

    std::vector<RayPixel> pixels;

public:
    RayBuffer() {}
    RayBuffer(uint, uint);
    uint getWidth() { return WIDTH; }
    uint getHeight() { return HEIGHT; }
    uint64_t size() { return pixels.size(); }

    void setPixel(uint, uint, PX_Color);
    void setPixel(uint, uint, uint8_t, uint8_t, uint8_t);
    RayPixel getPixel(uint, uint);
};

#endif // _RAYBUFFER_H_
