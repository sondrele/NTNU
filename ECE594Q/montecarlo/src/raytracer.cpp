#include "raytracer.h"

RayTracer::RayTracer() {
    WIDTH = 0;
    HEIGHT = 0;
    depth = 0;
    scene = NULL;
}

RayTracer::RayTracer(uint width, uint height, uint d) {
    WIDTH = width;
    HEIGHT = height;
    depth = d;
    numSamples = 1;
    scaleConst = 10000;
    fattjScale = 1.0f;
    buffer = RayBuffer(WIDTH, HEIGHT);
    scene = NULL;
    usingEnvMap = false;

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

void RayTracer::loadEnvMap(std::string name) {
    envMap.loadImage(name);
    usingEnvMap = true;
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

Ray RayTracer::computeRay(float x, float y) {
    Point_2D pt;
    pt.x = x * (1 / (float) WIDTH);
    pt.y = y * (1 / (float) HEIGHT);

    Vect dx = (horizontal()).linearMult(2 * pt.x - 1);
    Vect dy = (vertical()).linearMult(2 * pt.y - 1);

    Vect dir = imageCenter + dx + dy;
    dir.normalize();

    Ray r(getCameraPos(), dir);
    return r;
}

SColor RayTracer::calculateShadowScalar(Light *lt, Intersection &in, int d) {
    if (d <= 0) {
        return SColor(0, 0, 0);
    } else if (!in.hasIntersected()) {
        if (!usingEnvMap) {
            return SColor(0, 0, 0);
        } else {
            return envMap.getTexel(in.getDirection());
        }
    }

    int num_samples = 1;
    Vect p = lt->getPos();
    Vect ori = in.calculateIntersectionPoint();
    Vect dir;
    if (lt->getType() == DIRECTIONAL_LIGHT) {
        dir = lt->getDir().invert();
    } else if (lt->getType() == AREA_LIGHT) {
        num_samples = 4;
    } else {
        dir = p - ori;
        dir.normalize();
    }
    ori = ori + dir.linearMult(0.001f);

    float Sj = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        Ray shdw(ori, dir);

        Intersection ins = scene->intersects(shdw);
        Material *mat = ins.getMaterial();
        
        if (!ins.hasIntersected()) {
            // The point is in direct light
            Sj += 1.0f;
        } else if (mat->getTransparency() <= 0.00000001) {
            // The material is fully opaque
            Vect pos = ins.calculateIntersectionPoint();
            if (lt->getType() == DIRECTIONAL_LIGHT ||
                ori.euclideanDistance(pos) < ori.euclideanDistance(lt->getPos())) {
                // The ray intersects with an object before the light source
                return SColor(0, 0, 0);
            } else {
                // The ray intersects with an object behind the lightsource
                // or a direction light, thus fully in light
                Sj += 1;
            }
        } else { // The shape is transparent
            // Normalize the color for this material, and recursively trace for other
            // transparent objects
            SColor Cd = ins.getColor();
            float maxval = max(Cd.R(), max(Cd.G(), Cd.B()));
            Cd.R(Cd.R() / maxval); Cd.G(Cd.G() / maxval); Cd.B(Cd.B() / maxval);
            SColor Si = Cd * mat->getTransparency();
            Si = Si * calculateShadowScalar(lt, ins, d - 1);
            Sj += Si.R();
        }
    }

    Sj /= num_samples;
    return SColor(Sj, Sj, Sj);
}

float RayTracer::calculateFattj(Vect Pt, Light *l) {
    if (l->getType() == POINT_LIGHT) {
        float dist = Pt.euclideanDistance(l->getPos());
        return (float) min(1.0, fattjScale / (0.25 + 0.1 * dist + 0.01 * dist * dist));
    } else {
        return 1.0;
    }
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


SColor RayTracer::directIllumination(Light *lt, Intersection in, SColor Sj, float Fattj) {
    Vect Pt = in.calculateIntersectionPoint();
    Vect pos = lt->getPos();
    Material *mat = in.getMaterial();
    float kt = mat->getTransparency();
    SColor ks = mat->getSpecColor();
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
    return dirLight;
}
