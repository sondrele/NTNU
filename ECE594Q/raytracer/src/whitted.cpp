#include "whitted.h"


// Use ambColor given in the material paramter
SColor Whitted::AmbientLightning(float kt, SColor ka, SColor Cd) {
    assert(kt >= 0 && kt <= 1);

    return Cd.linearMult(ka).linearMult((1.0f - kt));
}

float Whitted::CalculateFattj(Vect Pt, Light *l) {
    if (l->getType() == POINT_LIGHT) {
        float dist = Pt.euclideanDistance(l->getPos());
        return (float) min(1.0, 1.0/(0.25 + 0.1 * dist + 0.01 * dist * dist));
    } else {
        return 1.0;
    }
}

SColor Whitted::Illumination(Light *lt, Intersection in, SColor Sj) {
    Vect Pt = in.calculateIntersectionPoint();
    Vect pos = lt->getPos();
    Material *mat = in.getMaterial();
    float kt = mat->getTransparency();
    SColor ks = mat->getSpecColor();
    SColor Cd = mat->getDiffColor();
    float q = mat->getShininess() * 128;
    float Fattj = Whitted::CalculateFattj(Pt, lt);
    SColor Ij = lt->getIntensity();
    
    SColor dirLight = Ij.linearMult(Sj).linearMult(Fattj);
    
    Vect Dj;
    if (lt->getType() == DIRECTIONAL_LIGHT) {
        Dj = lt->getDir().invert();
    } else {
        Dj = pos - Pt;
        Dj.normalize();
    }
    Vect N = in.calculateSurfaceNormal();
    SColor diffuseLight = Whitted::DiffuseLightning(kt, Cd, N, Dj);

    Vect V = in.getDirection().linearMult(-1);
    SColor specLight = Whitted::SpecularLightning(q, ks, N, Dj, V);

    dirLight = dirLight.linearMult(diffuseLight + specLight);
    // dirLight = dirLight.linearMult(Cd);

    // SColor Q = N * N.dotProduct(Dj);
    // SColor Rj = Q.linearMult(2) - Dj;
    return dirLight;
}

SColor Whitted::DiffuseLightning(float kt, SColor Cd, Vect N, Vect Dj) {
    float a = (1.0f - kt);
    float b = max(0.0f, N.dotProduct(Dj));

    // TODO: Flip normal if the ray is inside a transparent object
    return Cd.linearMult(a * b);
}

SColor Whitted::SpecularLightning(float q, SColor ks, Vect N, Vect Dj, Vect V) {
    float t;
    t = N.dotProduct(Dj); 
    Vect Q = N.linearMult(t);
    Vect Rj = Q.linearMult(2);
    Rj = Rj - Dj;
    t = Rj.dotProduct(V);
    t = max(t, 0.0f);
    assert(t >= 0);

    float f = pow(t, q);
    return ks.linearMult(f);
}

// SColor Whitted::Reflection(float ks, SColor Ls) {
//     assert(ks >= 0 && ks <= 1);
//     return Ls.linearMult(ks);
// }

// SColor Whitted::Refraction(float kt, SColor Lt) {
//     return Lt.linearMult(kt);
// }
