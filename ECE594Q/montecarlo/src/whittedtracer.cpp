#include "raytracer.h"

WhittedTracer::WhittedTracer(uint width, uint height, uint d)
: RayTracer(width, height, d) {

}

WhittedTracer::~WhittedTracer() {

}



SColor WhittedTracer::ambientLightning(float kt, SColor ka, SColor Cd) {
    return Cd.linearMult(ka).linearMult((1.0f - kt));
}

SColor WhittedTracer::shadeIntersection(Intersection in, int d) {
    if (d <= 0) {
        return SColor(0, 0, 0);
    } else if (!in.hasIntersected()) {
        if (!usingEnvMap) {
            return SColor(0, 0, 0);
        } else {
            return envMap.getTexel(in.getDirection());
        }
    }

    SColor shade(0, 0, 0);

    Material *mat = in.getMaterial();
    float kt = mat->getTransparency();
    SColor ks = mat->getSpecColor();
    SColor ka = mat->getAmbColor();
    Vect Pt = in.calculateIntersectionPoint();
    SColor Cd = in.getColor();

    SColor ambLight = ambientLightning(kt, ka, Cd);

    std::vector<Light *> lts = scene->getLights();
    for (uint i = 0; i < lts.size(); i++) {
        Light *l = lts.at(i);
        float Fattj = calculateFattj(Pt, l);
        if (Fattj > 0) {
            SColor Sj = calculateShadowScalar(l, in, (int) depth);
            shade = shade + directIllumination(l, in, Sj, Fattj);
        }
    }
    
    SColor reflection;
    if (ks.length() > 0) {
        Ray r = in.calculateReflection();
        Intersection rin = scene->intersects(r);
        reflection = shadeIntersection(rin, d - 1).linearMult(ks);
    }

    SColor refraction;
    if (kt > 0) {
        Ray r;
        if (in.calculateRefraction(r)) {
            Intersection rin = scene->intersects(r);
            refraction = shadeIntersection(rin, d - 1).linearMult(kt);
        } else {
            refraction = SColor(0, 0, 0);
        }
    }

    shade = ambLight + shade + reflection + refraction;

    return shade;
}

RayBuffer WhittedTracer::traceScene() {
    cout << "Tracing rays: " << WIDTH << "x" << HEIGHT << endl;
    cout << "numSamples = " << numSamples * numSamples << endl;
    cout << "depth = " << depth << endl;
    Progress p;
    p.setGoal((int) (HEIGHT * WIDTH));

    for (uint y = 0; y < HEIGHT; y++) {
        omp_set_num_threads(16);
        #pragma omp parallel for
        for (uint x = 0; x < WIDTH; x++) {
            // Loop over samples to get the right color
            float s = numSamples * numSamples;
            float R = 0, G = 0, B = 0;
            for (float dn = 0; dn < numSamples; dn += 1) {
                for (float dm = 0; dm < numSamples; dm += 1) {
                    float dx = dm / numSamples;
                    float dy = dn / numSamples;
                    Ray r = computeRay((float) (x + dx), (float) (y + dy));
                    Intersection in = scene->intersects(r);
                    SColor c = shadeIntersection(in, (int) depth);
                    R += c.R(); G += c.G(); B += c.B();
                }
            }
            R /= s; G /= s; B /= s;
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
