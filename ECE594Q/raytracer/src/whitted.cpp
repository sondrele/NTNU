#include "whitted.h"


// Use ambColor given in the material paramter
Vect Whitted::AmbientLightning(float kt, float ka, Vect Cd) {
    assert(kt >= 0 && kt <= 1);
    assert(ka >= 0 && ka <= 1);

    return Cd.linearMult(ka * (1.0f - kt));
}

Vect Whitted::Illumination(Light lt, Intersection in, float Sj) {
    Vect Pt = in.calculateIntersectionPoint();
    // float q = in.getShininess();
    // Vect N = in.getSurfaceNormal();

    // float Fattj = Whitted::CalculateFattj(Pt, lt);
    Vect Ij = lt.getIntensity();
    Vect dirLight = Ij.linearMult(Sj);
    // dirLight = dirLight + Whitted::DirectIllumination(Sj, Ij, Fattj);

    // Vect Dj = lt.getDir();
    // Vect diffLight = Whitted::DiffuseLightning(kt, Cd, N, Dj);

    // Vect Q = N * N.dotProduct(Dj);
    // Vect Rj = Q.linearMult(2) - Dj;
    // Vect specLight = Whitted::SpecularLightning(ks, Rj, V, q);

    return dirLight;
}

float Whitted::CalculateFattj(Vect Pt, Light l) {
    if (l.getType() == POINT_LIGHT) {
        float dist = Pt.euclideanDistance(l.getPos());
        return (float) min(1.0, 1.0/(0.25 + 0.1 * dist + 0.01 * dist * dist));
    } else {
        return 1.0;
    }
}

// Vect Whitted::DirectIllumination(Vect Sj, Vect Ij, float Fattj) {
//     return (Sj.dotProduct(Ij)).linearMult(Fattj);
// }

// Vect Whitted::DiffuseLightning(float kt, Vect Cd, Vect N, Vect D) {
//     float a = (1.0 - kt);
//     float b = max(0, N.dotProduct(D));

//     // TODO: Flip normal if the ray is inside a transparent object
//     return Cd.linearMult(a * b);
// }

// Vect Whitted::SpecularLightning(float ks, Vect Rj, Vect V, float q) {̈́
//     float a = Rj.dotProduct(V);
//     return ks * pow(a, q);
// }

// Vect Whitted::Reflection(float ks, Vect Ls) {
//     assert(ks >= 0 && ks <= 1);
//     return Ls.linearMult(ks);
// }

// Vect Whitted::Refraction(float kt, Vect Lt) {
//     return Lt.linearMult(kt);
// }
