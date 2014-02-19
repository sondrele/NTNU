#ifndef _WHITTED_H_
#define _WHITTED_H_ 

#include <cmath>
#include <cassert>

#include "Matrix.h"
#include "ray.h"
#include "rayscene.h"
#include "rayscene_shapes.h"

/*
L           radiance of the final ray (not radiometrically correct), calculated by your
            renderer for every pixel of the image (3-channel vector)
kt          scalar transmission coefficient of the first intersected object, ranging
            from 0 (fully opaque) to 1 (fully transparent)
ka          scalar ambient term (0 - 1) specifying the fraction of the diffuse color
            when not directly illuminated by a light source
Cd          diffuse reflection coefficient, indicates how much light is diffusively
            reflected for each color channel of the light source (i.e., the
            “base color” of the surface)
n           number of light sources in the scene
Sj          shadow factor vector indicating whether the current point can “see” the
            j th light source or not, computed by tracing a shadow ray to the light
            source (not necessarily binary because of transparency – see Sec. 3)
Ij          “intensity” color of the j th light source (three channel vector)
fattj       attenuation factor that reduces the contribution from the j th light
            source as the point gets farther away from it (see Sec. 3)
N           surface normal at the point of intersection (normalized)
Dj          the direction of the j th light source (normalized)
ks          scalar specular coefficient for the object indicating how reflective
            it is (0 - 1)
Rj          reflection vector for the j th light source for the Phong model
            (see Sec. 3.2)
V           flipped direction of the incident ray on the surface (normalized)
q           sets the “shininess” of the object by specifying how tight the
            specular highlight should be (if q < 0 we have more diffuse appearance,
            large q produces tight highlights)
Ls          radiance of the reflected ray, calculated by recursively tracing a ray
            from the intersection point and recomputing this equation
Lt          radiance of the refracted ray, also calculated recursively

*/

class Whitted {
public:
    static SColor Illumination(Light *, Intersection, SColor, float);

    static SColor AmbientLightning(float, SColor, SColor);
    static float CalculateFattj(Vect, Light *);
    static SColor DirectIllumination(SColor, SColor, float);
    static SColor DiffuseLightning(float, SColor, Vect, Vect);
    static SColor SpecularLightning(float, SColor, Vect, Vect, Vect);

    static SColor Reflection(float, SColor);
    static SColor Refraction(float, SColor);

    static float GetOutgoingRads(float, float, float);
};

#endif // _WHITTED_H_