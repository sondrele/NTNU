#include "raytracer.h"

#ifndef _WINDOWS
#include "omp.h"
#endif  // _WINDOWS

RayTracer::RayTracer(uint width, uint height, uint d) {
    WIDTH = width;
    HEIGHT = height;
    depth = d;
    scaleConst = 10000;
    buffer = RayBuffer(WIDTH, HEIGHT);
    scene = NULL;
    M = 1;

    // Set standard camera properties
    camera.setPos(Vect(0, 0, 0));
    camera.setVerticalFOV((float)M_PI / 2.0f);
    camera.setViewDir(Vect(0, 0, -1));
    camera.setOrthoUp(Vect(0, 1, 0));

    calculateImagePlane();
}

RayTracer::~RayTracer() {
    if (scene != NULL) {
        delete scene;
    }
}

void RayTracer::setCamera(Camera c) {
    camera = c;
    calculateImagePlane();
}

void RayTracer::setCameraPos(Vect camPos) {
    camera.setPos(camPos);
    calculateImagePlane();
}

void RayTracer::setViewDirection(Vect viewDir) {
    camera.setViewDir(viewDir);
    calculateImagePlane();
}

void RayTracer::setOrthogonalUp(Vect orthoUp) {
    camera.setOrthoUp(orthoUp);
    calculateImagePlane();
}

void RayTracer::calculateImagePlane() {
    Vect viewDir = camera.getViewDir();
    Vect orthoUp = camera.getOrthoUp();
    Vect cameraPos = camera.getPos();

    parallelRight = viewDir.crossProduct(orthoUp);
    parallelUp = parallelRight.crossProduct(viewDir);
    parallelRight.normalize();
    parallelUp.normalize();

    imageCenter = cameraPos + viewDir.linearMult(scaleConst);
}

Vect RayTracer::getImageCenter() {
    return imageCenter;
}

double RayTracer::getHorizontalFOV() {
    double horiFov = ((float) WIDTH / (float) HEIGHT) * getVerticalFOV();
    if (horiFov >= M_PI)
        throw "FOV too large";
    return horiFov;
}

Vect RayTracer::vertical() {
    float f = (float) tan(getVerticalFOV() / 2) * scaleConst;
    Vect vert = parallelUp.linearMult(f);
    return vert;
}

Vect RayTracer::horizontal() {
    float f = (float) tan(getHorizontalFOV() / 2) * scaleConst;
    Vect hori = parallelRight.linearMult(f);
    return hori;
}

Point_2D RayTracer::computePoint(uint x, uint y) {
    Point_2D pt;
    pt.x = (float)(x + 0.5) * (1 / (float) WIDTH);
    pt.y = (float)(y + 0.5) * (1 / (float) HEIGHT);
    return pt;
}

Vect RayTracer::computeDirection(uint x, uint y) {
    Point_2D p = computePoint(x, y);
    Vect dx = (horizontal()).linearMult(2 * p.x - 1);
    Vect dy = (vertical()).linearMult(2 * p.y - 1);

    Vect dir = imageCenter + dx + dy;
    dir.normalize();
    return dir;
}

Ray RayTracer::computeRay(uint x, uint y) {
    Vect dir = computeDirection(x, y);
    Ray r(getCameraPos(), dir);
    return r;
}

Ray RayTracer::computeMonteCarloRay(float x, float y) {
    Point_2D pt;
    pt.x = x * (1 / (float) WIDTH);
    pt.y = y * (1 / (float) HEIGHT);

    Vect dx = (horizontal()).linearMult(2 * pt.x - 1);
    Vect dy = (vertical()).linearMult(2 * pt.y - 1);

    Vect dir = imageCenter + dx + dy;
    dir.normalize();

    // Vect dir = computeDirection(x, y);
    Ray r(getCameraPos(), dir);
    return r;
}

SColor RayTracer::calculateShadowScalar(Light &lt, Intersection &in, int d) {
    if (d < 0) {
        return SColor(0, 0, 0);   
    }

    Vect p = lt.getPos();
    Vect ori = in.calculateIntersectionPoint();// + in.calculateSurfaceNormal().linearMult(0.0001f);
    Vect dir;
    if (lt.getType() == DIRECTIONAL_LIGHT) {
        dir = lt.getDir().invert();
    } else {
        dir = p - ori;
        dir.normalize();
    }
    ori = ori + dir.linearMult(0.001f);
    Ray shdw(ori, dir);

    Intersection ins = scene->intersectsWithBVHTree(shdw);
    Material *mat = ins.getMaterial();
    
    if (!ins.hasIntersected()) {
        // The point is in direct light
        return SColor(1, 1, 1);
    } else if (mat->getTransparency() <= 0.00000001) {
        // The material is fully opaque
        Vect pos = ins.calculateIntersectionPoint();
        if (lt.getType() == DIRECTIONAL_LIGHT ||
            ori.euclideanDistance(pos) < ori.euclideanDistance(lt.getPos())) {
            // The ray intersects with an object before the light source
            return SColor(0, 0, 0);
        } else {
            // The ray intersects with an object behind the lightsource
            // or a direction light, thus fully in light
            return SColor(1, 1, 1);
        }
    } else { // The shape is transparent
        // Normalize the color for this material, and recursively trace for other
        // transparent objects
        // SColor Cd = mat->getDiffColor();
        SColor Cd = ins.getColor();
        float maxval = max(Cd.R(), max(Cd.G(), Cd.B()));
        Cd.R(Cd.R() / maxval); Cd.G(Cd.G() / maxval); Cd.B(Cd.B() / maxval);
        SColor Si = Cd.linearMult(mat->getTransparency());
        return Si.linearMult(calculateShadowScalar(lt, ins, d - 1));
    }
}

SColor RayTracer::shadeIntersection(Intersection in, int d) {
    if (d <= 0 || !in.hasIntersected()) {
        // terminate recursion
        return SColor(0, 0, 0);
        }

    SColor shade(0, 0, 0);

    Material *mat = in.getMaterial();
    float kt = mat->getTransparency();
    SColor ks = mat->getSpecColor();
    SColor ka = mat->getAmbColor();
    // SColor Cd = mat->getDiffColor();
    Vect Pt = in.calculateIntersectionPoint();
    SColor Cd = in.getColor();

    SColor ambLight = ambientLightning(kt, ka, Cd);

    std::vector<Light *> lts = scene->getLights();
    for (uint i = 0; i < lts.size(); i++) {
        Light *l = lts.at(i);
        float Fattj = calculateFattj(Pt, l);
        if (Fattj > 0) {
            SColor Sj = calculateShadowScalar(*l, in, (int) depth);
            shade = shade + directIllumination(l, in, Sj, Fattj);
        }
    }
    
    SColor reflection;
    if (ks.length() > 0) {
        Ray r = in.calculateReflection();
        Intersection rin = scene->intersectsWithBVHTree(r);
        reflection = shadeIntersection(rin, d - 1).linearMult(ks);
    }

    SColor refraction;
    if (kt > 0) {
        Ray r;
        if (in.calculateRefraction(r)) {
            Intersection rin = scene->intersectsWithBVHTree(r);
            refraction = shadeIntersection(rin, d - 1).linearMult(kt);
        } else {
            refraction =  SColor(0, 0, 0);
        }
    }

    shade = ambLight + shade + reflection + refraction;

    return shade;
}

float RayTracer::calculateFattj(Vect Pt, Light *l) {
    if (l->getType() == POINT_LIGHT) {
        float dist = Pt.euclideanDistance(l->getPos());
        return (float) min(1.0, 1.0 / (0.25 + 0.1 * dist + 0.01 * dist * dist));
    } else {
        return 1.0;
    }
}

SColor RayTracer::ambientLightning(float kt, SColor ka, SColor Cd) {
    // assert(kt >= 0 && kt <= 1);
    return Cd.linearMult(ka).linearMult((1.0f - kt));
}

SColor RayTracer::directIllumination(Light *lt, Intersection in, SColor Sj, float Fattj) {
    Vect Pt = in.calculateIntersectionPoint();
    Vect pos = lt->getPos();
    Material *mat = in.getMaterial();
    float kt = mat->getTransparency();
    SColor ks = mat->getSpecColor();
    // SColor Cd = mat->getDiffColor();
    SColor Cd = in.getColor();
    float q = mat->getShininess() * 128;
    SColor Ij = lt->getIntensity();
    
    SColor dirLight = Ij.linearMult(Sj).linearMult(Fattj);
    
    Vect Dj;
    if (lt->getType() == DIRECTIONAL_LIGHT) {
        Dj = lt->getDir().invert();
    } else {
        Dj = pos - Pt;
        Dj.normalize();
    }
    Vect Norm = in.calculateSurfaceNormal();
    SColor diffuseLight = diffuseLightning(kt, Cd, Norm, Dj);

    Vect V = in.getDirection().linearMult(-1);
    SColor specLight = specularLightning(q, ks, Norm, Dj, V);

    dirLight = dirLight.linearMult(diffuseLight + specLight);
    // dirLight = dirLight.linearMult(Cd);

    // SColor Q = N * N.dotProduct(Dj);
    // SColor Rj = Q.linearMult(2) - Dj;
    return dirLight;
}

SColor RayTracer::diffuseLightning(float kt, SColor Cd, Vect Norm, Vect Dj) {
    float a = (1.0f - kt);
    float b = max(0.0f, Norm.dotProduct(Dj));

    // TODO: Flip normal if the ray is inside a transparent object
    return Cd.linearMult(a * b);
}

SColor RayTracer::specularLightning(float q, SColor ks, Vect Norm, Vect Dj, Vect V) {
    float t = Norm.dotProduct(Dj); 
    Vect Q = Norm.linearMult(t);
    Vect Rj = Q.linearMult(2);
    Rj = Rj - Dj;
    t = Rj.dotProduct(V);
    t = max(t, 0.0f);

    float f = pow(t, q);
    return ks.linearMult(f);
}

// bool RayTracer::russianRoulette(SColor refl, float &survivorMult) {
//     double p = max(refl.R(), max(refl.G(), refl.B()));
//     survivorMult = 1.0 / p;
//     if (Rand::Random() > p) 
//         return true;
//     return false;
// }

RayBuffer RayTracer::traceRays() {
    for (uint y = 0; y < HEIGHT; y++) {
        // omp_set_num_threads(16);
        // #pragma omp parallel for
        for (uint x = 0; x < WIDTH; x++) {
            Ray r = computeRay(x, y);
            Intersection in = scene->intersectsWithBVHTree(r);
            SColor c = shadeIntersection(in, (int) depth);
            PX_Color color;
            color.R = (uint8_t) (255 * c.R());
            color.G = (uint8_t) (255 * c.G());
            color.B = (uint8_t) (255 * c.B());
            buffer.setPixel(x, y, color);
        }
    }
    return buffer;
}

RayBuffer RayTracer::traceRaysWithAntiAliasing() {
    cout << "Tracing rays[" << WIDTH << "][" << HEIGHT << "]" << endl;
    cout << "m = " << M << ", n = " << M << endl;
    cout << "d = " << depth << endl;
    for (uint y = 0; y < HEIGHT; y++) {
        // omp_set_num_threads(16);
        // #pragma omp parallel for
        for (uint x = 0; x < WIDTH; x++) {
            // Loop over samples to get the right color
            float R = 0, G = 0, B = 0;
            for (float dn = 0; dn < M; dn += 1) {
                for (float dm = 0; dm < M; dm += 1) {
                    float dx = dm / M;
                    float dy = dn / M;
                    Ray r = computeMonteCarloRay((float) x + dx, (float) y + dy);
                    Intersection in = scene->intersectsWithBVHTree(r);
                    SColor c = shadeIntersection(in, (int) depth);
                    R += c.R(); G += c.G(); B += c.B();
                }
            }
            R /= M * M; G /= M * M; B /= M * M;
            PX_Color color;
            color.R = (uint8_t) (255 * R);
            color.G = (uint8_t) (255 * G);
            color.B = (uint8_t) (255 * B);
            buffer.setPixel(x, y, color);
        }
    }
    return buffer;
}
