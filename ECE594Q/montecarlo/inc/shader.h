#ifndef _SHADER_H
#define _SHADER_H

#include "material.h"
#include "texture.h"

/**********************
* Color Shaders
**********************/
class CShader {
protected:
    Material *mat;
    Texture *tex;

public:
    CShader();
    virtual ~CShader() {}

    void setMaterial(Material *m);
    Material * getMaterial();
    void setTexture(Texture *t);
    Texture * getTexture();
    virtual SColor getColor(Point_2D);
};

class CheckCShader : public CShader {
public:
    virtual ~CheckCShader() {}
    virtual SColor getColor(Point_2D);
};

/**********************
* Intersection Shaders
**********************/
class IShader {
public:
    IShader() {}
    virtual ~IShader() {}

    virtual bool hasIntersected(Point_2D);
};

class CheckIShader : public IShader {
public:
    virtual ~CheckIShader() {}
    virtual bool hasIntersected(Point_2D);
};

#endif // _SHADER_H