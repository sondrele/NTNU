#ifndef _RAYBUFFER_H_
#define _RAYBUFFER_H_

#include <stdint.h>
#include <stdlib.h>
#include <vector>

#define PX(x, y) (y * WIDTH + x)

typedef struct {
    unsigned char R;
    unsigned char G;
    unsigned char B;
} PX_Color;

class RayPixel {
private:
    uint X;
    uint Y;
    float opacity;
    PX_Color color;

public:
    RayPixel(uint, uint);
    uint getX() { return X; };
    uint getY() { return Y; };
    void setColor(PX_Color);
    void setColor(unsigned char, unsigned char, unsigned char);
    PX_Color getColor() { return color; };
};

class RayBuffer {
private:
    uint WIDTH;
    uint HEIGHT;

    std::vector<RayPixel> pixels;

public:
    RayBuffer(uint, uint);
    uint getWidth() { return WIDTH; };
    uint getHeight() { return HEIGHT; };
    uint64_t size() { return pixels.size(); };

    void setPixel(uint, uint, PX_Color);
    void setPixel(uint, uint, unsigned char, unsigned char, unsigned char);
    RayPixel getPixel(uint, uint);
};

#endif // _RAYBUFFER_H_