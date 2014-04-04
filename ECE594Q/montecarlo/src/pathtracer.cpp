#include "raytracer.h"

PathTracer::PathTracer(uint width, uint height, uint d)
: RayTracer(width, height, d) {
    bidirectional = false;
}

PathTracer::~PathTracer() {

}

Ray PathTracer::computeaRayFromLightSource(Light *l) {
    Ray r;
    if (l->getType() != DIRECTIONAL_LIGHT) {
        r.setOrigin(l->getPos());
        r.setDirection(Rand::RandomVect());
    }
    return r;
}

Light * PathTracer::pickRandomLight() {
    std::vector<Light *> lts = scene->getLights();
    if (!lts.empty()) {
        uint pos = (uint) ((double) lts.size() * Rand::Random());
        return lts.at(pos);
    }
    return NULL;
}

float PathTracer::fattj(Vect Pt, Vect pos) {
    float dist = Pt.euclideanDistance(pos);
    return (float) min(1.0, fattjScale / (0.25 + 0.1 * dist + 0.01 * dist * dist));
}

Vect PathTracer::specularSampleUpperHemisphere(Intersection &ins) {
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

SColor PathTracer::shootRayFromLightSource(Light *l, Vect &intersectionPt, int s) {
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

SColor PathTracer::connectPaths(Vect &gatheringPt, Vect &shootingPt, SColor &shootingEmmittance) {
    Ray r;
    Vect dir = shootingPt - gatheringPt;
    dir.normalize();
    r.setDirection(dir);
    r.setOrigin(gatheringPt + dir * 0.001f);
    Intersection in = scene->intersects(r);

    // If true, then the paths can be connected
    if (in.hasIntersected() && 
        shootingPt.euclideanDistance(in.calculateIntersectionPoint()) < 0.001) {
        return shootingEmmittance * fattj(gatheringPt, shootingPt);
    } else {
        return SColor();
    }
}

SColor PathTracer::shadeIntersectionPoint(Intersection &in, Vect &intersectionPt,
    int &s, int t, bool sampleAreaLights)
{
    SColor shade;
    intersectionPt = in.calculateIntersectionPoint();
    
    if (t < 0) {
        if (!bidirectional) {
            return shade;
        }

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
    
    Vect N = in.calculateSurfaceNormal();
    Light *l = pickRandomLight();

    if (l != NULL) {
        float Fattj = calculateFattj(intersectionPt, l);
        if (Fattj > 0) {
            SColor Sj;
            if (sampleAreaLights)
                Sj = calculateShadowScalar(l, in, (int) depth, 1);
            else
                Sj = calculateShadowScalar(l, in, (int) depth, 10);
            shade = shade + directIllumination(l, in, Sj, Fattj);
        }
    }

    SColor Cd = in.getColor();

    Ray r;
    r.setOrigin(intersectionPt + N * 0.0001);
    r.setDirection(specularSampleUpperHemisphere(in));

    in = scene->intersects(r);
    if (in.hasIntersected()) {
        shade = shade + Cd * shadeIntersectionPoint(in, intersectionPt, s, t - 1, false);
    }
    
    return shade;
}

bool PathTracer::traceRayFromCamera(uint x, uint y, SColor &shade, int s, int t) {
    Ray r = computeRay((float) x, (float) y);
    Intersection in = scene->intersects(r);

    if (in.hasIntersected()) {
        Vect intersectionPt;
        shade = shadeIntersectionPoint(in, intersectionPt, s, t, true);
        return true;
    } else {
        return false;
    }
}

RayBuffer PathTracer::traceScene() {
    if (bidirectional)
        cout << "Tracing bidirectional paths: " << WIDTH << "x" << HEIGHT << endl;
    else
        cout << "Tracing paths: " << WIDTH << "x" << HEIGHT << endl;
    cout << "depth = " << depth << endl;
    cout << "numSamples = " << numSamples << endl;
    Progress p;
    p.setGoal((int) (HEIGHT * WIDTH));

    for (uint y = 0; y < HEIGHT; y++) {
        omp_set_num_threads(16);
        #pragma omp parallel for
        for (uint x = 0; x < WIDTH; x++) {

            bool intersectsScene;
            int s = 0, t = depth;
            float R = 0, G = 0, B = 0;
            for (int ns = 0; ns < numSamples; ns++) {
                SColor shade;

                if (bidirectional) {
                    s = (int) (Rand::Random() * depth);
                    t = (int) (Rand::Random() * depth);
                }

                intersectsScene = traceRayFromCamera(x, y, shade, s, t);
                R += shade.R();
                G += shade.G();
                B += shade.B();

                if (!intersectsScene) {
                    break;
                }

            }

            if (intersectsScene) {
                R /= (float) numSamples;
                G /= (float) numSamples;
                B /= (float) numSamples;
            }

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
