#ifndef _MATERIAL_H_
#define _MATERIAL_H_

#include <sstream>
#include <iostream>

#include "Matrix.h"
#include "scene_io.h"

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

    bool isReflective();
    bool isRefractive();

    friend ostream& operator <<(ostream &, const Material &);
};

#endif // _MATERIAL_H_