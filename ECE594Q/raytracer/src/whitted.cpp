#include "whitted.h"


// Use ambColor given in the material paramter
SColor Whitted::AmbientLightning(float kt, float ka, SColor Cd) {
    assert(kt >= 0 && kt <= 1);
    assert(ka >= 0 && ka <= 1);

    return Cd.linearMult(ka * (1.0f - kt));
}

SColor Whitted::Illumination(Light lt, Intersection in, float Sj) {
    SColor Pt = in.calculateIntersectionPoint();
    Material *mat = in.getMaterial();
    SColor Cd = mat->getDiffColor();
    // float q = in.getShininess();
    // SColor N = in.getSurfaceNormal();

    // float Fattj = Whitted::CalculateFattj(Pt, lt);
    SColor Ij = lt.getIntensity();
    
    SColor dirLight = Ij.linearMult(Sj);
    dirLight = dirLight.mult(Cd);
    // dirLight = dirLight + Whitted::DirectIllumination(Sj, Ij, Fattj);

    // SColor Dj = lt.getDir();
    // SColor diffLight = Whitted::DiffuseLightning(kt, Cd, N, Dj);

    // SColor Q = N * N.dotProduct(Dj);
    // SColor Rj = Q.linearMult(2) - Dj;
    // SColor specLight = Whitted::SpecularLightning(ks, Rj, V, q);
    return dirLight;
}

float Whitted::CalculateFattj(SColor Pt, Light l) {
    if (l.getType() == POINT_LIGHT) {
        float dist = Pt.euclideanDistance(l.getPos());
        return (float) min(1.0, 1.0/(0.25 + 0.1 * dist + 0.01 * dist * dist));
    } else {
        return 1.0;
    }
}

// SColor Whitted::DirectIllumination(SColor Sj, SColor Ij, float Fattj) {
//     return (Sj.dotProduct(Ij)).linearMult(Fattj);
// }

// SColor Whitted::DiffuseLightning(float kt, SColor Cd, SColor N, SColor D) {
//     float a = (1.0 - kt);
//     float b = max(0, N.dotProduct(D));

//     // TODO: Flip normal if the ray is inside a transparent object
//     return Cd.linearMult(a * b);
// }

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
