#include "shader.h"

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

bool IShader::hasIntersected(Point_2D pt) {
    (void) pt;
    return true;
}

bool CheckIShader::hasIntersected(Point_2D pt) {    
    // int x = (int) (8 * pt.x);
    // int y = (int) (8 * pt.y);
    // x = x % 2; y = y % 2;
    
    if (pt.x < 0.5) {
        cout << "no" << endl;
        return false;
    } else {
        return true;
    }
}
