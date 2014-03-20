#include "raytracer.h"

BiPathTracer::BiPathTracer(uint width, uint height, uint d)
: RayTracer(width, height, d) {
}

BiPathTracer::~BiPathTracer() {

}

Ray BiPathTracer::computeaRayFromLightSource(Light *l) {
    Ray r;
    if (l->getType() == POINT_LIGHT) {
        r.setOrigin(l->getPos());
        r.setDirection(Rand::RandomVect());
    }
    return r;
}

Light * BiPathTracer::pickRandomLight() {
    std::vector<Light *> lts = scene->getLights();
    if (!lts.empty()) {
        uint pos = (uint) ((double) lts.size() * Rand::Random());
        return lts.at(pos);
    }
    return NULL;
}

float BiPathTracer::fattj(Vect Pt, Vect pos) {
    float dist = Pt.euclideanDistance(pos);
    return (float) min(1.0, fattjScale / (0.25 + 0.1 * dist + 0.01 * dist * dist));
}

Vect BiPathTracer::specularSampleUpperHemisphere(Intersection &ins) {
    float specular = ins.specColor().length();
    float x = (1 - specular) * (M_PI - 2 * M_PI * (float) Rand::Random());
    float y = (1 - specular) * (M_PI - 2 * M_PI * (float) Rand::Random());

    Vect dir = ins.calculateReflection().getDirection();
    Vect_h dir_h(dir.getX(), dir.getY(), dir.getZ());
    Vect_h::Rotate(dir_h, 'x', x);
    Vect_h::Rotate(dir_h, 'y', y);

    Vect randDir(dir_h.getX(), dir_h.getY(), dir_h.getZ());
    randDir.normalize();
    if (randDir.dotProduct(ins.calculateSurfaceNormal()) < 0)
        return randDir.invert();
    return randDir;
}

SColor BiPathTracer::shootRayFromLightSource(Light *l, Vect &intersectionPt, int s) {
    Ray r = computeaRayFromLightSource(l);
    Intersection in = scene->intersects(r);
    SColor emmittance;
    if (in.hasIntersected()) {
        emmittance = in.getColor() * l->getIntensity();

        intersectionPt = in.calculateIntersectionPoint();
        Vect N = in.calculateSurfaceNormal();
        for (int i = 1; i < s; i++) {
            r.setOrigin(intersectionPt + N * 0.0001);
            r.setDirection(specularSampleUpperHemisphere(in));

            in = scene->intersects(r);
            if (in.hasIntersected()) {
                intersectionPt = in.calculateIntersectionPoint();
                N = in.calculateSurfaceNormal();
                emmittance = emmittance + emmittance * in.getColor();
            } else {
                return emmittance;
            }
        }
    }
    return emmittance;
}

SColor BiPathTracer::connectPaths(Vect &gatheringPt, Vect &shootingPt, SColor &shootingEmmittance) {
    Ray r(gatheringPt, shootingPt);
    Intersection in = scene->intersects(r);

    // If true, then the paths can be connected
    if (in.hasIntersected() && 
        shootingPt.euclideanDistance(in.calculateIntersectionPoint()) < 0.0001) {
        return shootingEmmittance * fattj(gatheringPt, shootingPt);
    } else {
        return SColor();
    }
}

SColor BiPathTracer::shadeIntersectionPoint(Intersection &in, Vect &intersectionPt, int &s, int t) {
    SColor shade;
    
    if (t < 0) {
        Vect shootingInsPt;
        Light *l = pickRandomLight();
        SColor shootingEmmittance = shootRayFromLightSource(l, shootingInsPt, s);
        return connectPaths(intersectionPt, shootingInsPt, shootingEmmittance);
    } else if (!in.hasIntersected()) {
        if (!usingEnvMap) {
            return shade;
        } else {
            return envMap.getTexel(in.getDirection());
        }
    }

    Material *mat = in.getMaterial();
    shade = mat->getAmbColor();
    
    intersectionPt = in.calculateIntersectionPoint();
    Vect N = in.calculateSurfaceNormal();
    Light *l = pickRandomLight();

    if (l != NULL) {
        float Fattj = calculateFattj(intersectionPt, l);
        if (Fattj > 0) {
            SColor Sj = calculateShadowScalar(l, in, (int) depth);
            shade = shade + directIllumination(l, in, Sj, Fattj);
        }
    }

    SColor Cd = in.getColor();

    Ray r;
    r.setOrigin(intersectionPt + N * 0.0001);
    r.setDirection(specularSampleUpperHemisphere(in));

    in = scene->intersects(r);
    if (in.hasIntersected()) {
        shade = shade + Cd * shadeIntersectionPoint(in, intersectionPt, s, t - 1);
    }
    
    return shade;
}

SColor BiPathTracer::traceRayFromCamera(uint x, uint y, Vect &intersectionPt, int s, int t) {
    Ray r = computeRay((float) x, (float) y);
    Intersection in = scene->intersects(r);

    return shadeIntersectionPoint(in, intersectionPt, s, t);
}

RayBuffer BiPathTracer::traceScene() {
    cout << "Tracing bidirectional paths: " << WIDTH << "x" << HEIGHT << endl;
    cout << "depth = " << depth << endl;
    cout << "numSamples = " << numSamples << endl;
    Progress p;
    p.setGoal((int) (HEIGHT * WIDTH));

    for (uint y = 0; y < HEIGHT; y++) {
        omp_set_num_threads(16);
        #pragma omp parallel for
        for (uint x = 0; x < WIDTH; x++) {

            float R = 0, G = 0, B = 0;
            for (int ns = 0; ns < numSamples; ns++) {
                int s = (int) ((depth + 1) * Rand::Random());
                int t = depth - s;
                Vect gatheringInsPt;
                SColor shade = traceRayFromCamera(x, y, gatheringInsPt, s, t);

                R += shade.R();
                G += shade.G();
                B += shade.B();

            }
            R /= (float) numSamples;
            G /= (float) numSamples;
            B /= (float) numSamples;

            PX_Color color;
            color.R = (uint8_t) (255 * R);
            color.G = (uint8_t) (255 * G);
            color.B = (uint8_t) (255 * B);
            buffer.setPixel(x, y, color);

            #pragma omp critical
            {
                p.tick();
            }
        }
    }
    return buffer;
}

bool BiPathTracer::russianRoulette(SColor refl) {
    float p = max(refl.R(), max(refl.G(), refl.B()));
    if (Rand::Random() > p) 
        return true;
    return false;
}
