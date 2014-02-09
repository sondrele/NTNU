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

SColor Whitted::Illumination(Light *lt, Intersection in, float Sj) {
    Vect Pt = in.calculateIntersectionPoint();
    Vect pos = lt->getPos();
    Material *mat = in.getMaterial();
    float kt = mat->getTransparency();
    SColor Cd = mat->getDiffColor();
    // float q = in.getShininess();
    // SColor N = in.getSurfaceNormal();

    float Fattj = Whitted::CalculateFattj(Pt, lt);
    SColor Ij = lt->getIntensity();
    
    SColor dirLight = Ij.linearMult(Sj * Fattj);
    
    Vect Dj = pos - Pt;
    Dj.normalize();
    Vect N = in.calculateSurfaceNormal();
    SColor diffuseLight = Whitted::DiffuseLightning(kt, Cd, N, Dj);

    dirLight = dirLight.linearMult(diffuseLight);
    // dirLight = dirLight.linearMult(Cd);

    // SColor Q = N * N.dotProduct(Dj);
    // SColor Rj = Q.linearMult(2) - Dj;
    // SColor specLight = Whitted::SpecularLightning(ks, Rj, V, q);
    return dirLight;
}

SColor Whitted::DiffuseLightning(float kt, SColor Cd, Vect N, Vect D) {
    float a = (1.0f - kt);
    float b = max(0.0f, N.dotProduct(D));

    // TODO: Flip normal if the ray is inside a transparent object
    return Cd.linearMult(a * b);
}

// SColor Whitted::SpecularLightning(float ks, SColor Rj, SColor V, float q) {̈́
//     float a = Rj.dotProduct(V);
//     return ks * pow(a, q);
// }

// SColor Whitted::Reflection(float ks, SColor Ls) {
//     assert(ks >= 0 && ks <= 1);
//     return Ls.linearMult(ks);
// }

// SColor Whitted::Refraction(float kt, SColor Lt) {
//     return Lt.linearMult(kt);
// }
