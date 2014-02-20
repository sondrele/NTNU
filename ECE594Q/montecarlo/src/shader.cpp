#include "shader.h"

/****************************
* Color shader
****************************/
CShader::CShader() {
    mat = NULL;
    tex = NULL;
}

void CShader::setMaterial(Material *m) {
    mat = m;
}

Material * CShader::getMaterial() {
    return mat;
}

void CShader::setTexture(Texture *t) {
    tex = t;
}

Texture * CShader::getTexture() {
    return tex;
}

SColor CShader::getColor(Point_2D pt) {
    if (tex == NULL) {
        return mat->getDiffColor();
    } else {
        return tex->getTexel(pt.x, pt.y);
    }
}

/****************************
* Check Color shader
****************************/
SColor CheckCShader::getColor(Point_2D pt) {
    int x = (int) (8 * pt.x);
    int y = (int) (8 * pt.y);
    x = x % 2; y = y % 2;

    if ((x == 1 && y == 1) || (x == 0 && y == 0)) {
        return SColor();
    } else if (tex != NULL) {
        return tex->getTexel(pt.x, pt.y);
    } else {
        return mat->getDiffColor();
    }
}

/****************************
* Fun Color shader
****************************/
SColor FunCShader::getColor(Point_2D pt) {
    SColor color(1, 1, 0);
    color.R(pt.x * color.R());
    color.G(pt.y * color.G());
    return color;
}

/****************************
* Intersection shader
****************************/
bool IShader::hasIntersected(Point_2D pt) {
    (void) pt;
    return true;
}

/****************************
* Check Intersection shader
****************************/
bool CheckIShader::hasIntersected(Point_2D pt) {    
    int x = (int) (8 * pt.x);
    int y = (int) (8 * pt.y);
    x = x % 2; y = y % 2;
    // TODO: Fiks
    if ((x == 1 && y == 1) || (x == 0 && y == 0)) {
        return true;
    } else {
        return false;
    }
}
