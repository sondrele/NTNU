#include "raytracer.h"

RayTracer::RayTracer(uint width, uint height) {
    WIDTH = width;
    HEIGHT = height;
    depth = 1;
    scaleConst = 100;
    buffer = RayBuffer(WIDTH, HEIGHT);
    scene = NULL;

    camera.setPos(Vect(0, 0, 0));
    camera.setVerticalFOV((float)M_PI / 2.0f);
    camera.setViewDir(Vect(0, 0, -1));
    camera.setOrthoUp(Vect(0, 1, 0));

    calculateImagePlane();
}

RayTracer::RayTracer(uint width, uint height, uint d) {
    WIDTH = width;
    HEIGHT = height;
    depth = d;
    scaleConst = 100;
    buffer = RayBuffer(WIDTH, HEIGHT);
    scene = NULL;

    camera.setPos(Vect(0, 0, 0));
    camera.setVerticalFOV((float)M_PI / 2.0f);
    camera.setViewDir(Vect(0, 0, -1));
    camera.setOrthoUp(Vect(0, 1, 0));

    calculateImagePlane();
}

RayTracer::RayTracer(uint width, uint height, Vect viewDir, Vect orthoUp) {
    WIDTH = width;
    HEIGHT = height;
    depth = 1;
    scaleConst = 100;
    buffer = RayBuffer(WIDTH, HEIGHT);
    scene = NULL;

    camera.setPos(Vect(0, 0, 0));
    camera.setVerticalFOV((float)M_PI / 2.0f);
    camera.setViewDir(viewDir);
    camera.setOrthoUp(orthoUp);

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
    if (!(x < WIDTH && y < HEIGHT))
        throw "Coords out of bounds";

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
    // dir.normalize(); TODO: fix tests
    return dir;
}

Ray RayTracer::computeRay(uint x, uint y) {
    Vect dir = computeDirection(x, y);
    Ray r(getCameraPos(), dir);
    return r;
}

SColor RayTracer::calculateShadowScalar(Light &lt, Intersection &in) {
    // cout << in.toString() << endl;
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

    Intersection ins = scene->calculateRayIntersection(shdw);
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
        SColor Cd = mat->getDiffColor();
        float m = max(Cd.R(), max(Cd.G(), Cd.B()));
        Cd.R(Cd.R()/m); Cd.G(Cd.G()/m); Cd.B(Cd.B()/m);
        SColor Si = Cd.linearMult(mat->getTransparency());
        return Si.linearMult(calculateShadowScalar(lt, ins));
    }
}

SColor RayTracer::shadeIntersection(Intersection in, uint d) {
    if (d <= 0 || in.hasIntersected() == false) {
        // terminate recursion
        return SColor(0, 0, 0);
    }

    Vect shade(0, 0, 0);

    Material *mat = in.getMaterial();
    float kt = mat->getTransparency();
    SColor ks = mat->getSpecColor();
    SColor ka = mat->getAmbColor();
    SColor Cd = mat->getDiffColor();

    SColor ambLight = Whitted::AmbientLightning(kt, ka, Cd);

    std::vector<Light *> lts = scene->getLights();
    for (uint i = 0; i < lts.size(); i++) {
        Light *l = lts.at(i);

        SColor Sj = calculateShadowScalar(*l, in);
        shade = shade + Whitted::Illumination(l, in, Sj);
    }
    
    SColor reflection;
    if (ks.length() > 0) {
        Ray r = in.calculateReflection();
        Intersection rin = scene->calculateRayIntersection(r);
        reflection = shadeIntersection(rin, d-1).linearMult(ks);
    }

    SColor refraction;
    if (kt > 0) {
        Ray r = in.calculateRefraction();
        Intersection rin = scene->calculateRayIntersection(r);
        refraction = shadeIntersection(rin, d-1).linearMult(kt);
    }

    shade = ambLight + shade + reflection + refraction;

    return shade;
}

RayBuffer RayTracer::traceRays() {
    // #pragma omp parallel for
    for (uint y = 0; y < HEIGHT; y++) {
        // #pragma omp parallel for
        for (uint x = 0; x < WIDTH; x++) {
            Ray r = computeRay(x, y);
            Intersection in = scene->calculateRayIntersection(r);
            SColor c = shadeIntersection(in, depth);
            PX_Color color;
            color.R = (uint8_t) (255 * c.R());
            color.G = (uint8_t) (255 * c.G());
            color.B = (uint8_t) (255 * c.B());
            buffer.setPixel(x, y, color);
        }
    }
    return buffer;
}
