#include "raytracer.h"

PathTracer::PathTracer(uint width, uint height, uint d)
: RayTracer(width, height, d) {
}

PathTracer::~PathTracer() {

}

RayBuffer PathTracer::traceScene() {
    cout << "Tracing paths: " << WIDTH << "x" << HEIGHT << endl;
    cout << "depth = " << depth << endl;
    cout << "numSamples = " << numSamples << endl;
    Progress p;
    p.setGoal((int) (HEIGHT * WIDTH));

    for (uint y = 0; y < HEIGHT; y++) {
        omp_set_num_threads(16);
        #pragma omp parallel for
        for (uint x = 0; x < WIDTH; x++) {
            // Loop over samples to get the right color
            Ray r = computeRay((float) x, (float) y);
            Intersection in = scene->intersectsWithBVHTree(r);
            SColor c = shadeIntersectionPath(in, (int) depth);
            float R = c.R(), G = c.G(), B = c.B();

            if (in.hasIntersected()) {
                for (int s = 1; s < numSamples; s++) {
                    r = computeRay((float) x, (float) y);
                    in = scene->intersectsWithBVHTree(r);
                    c = shadeIntersectionPath(in, (int) depth);
                    R += c.R();
                    G += c.G();
                    B += c.B();
                }
                R /= (float) numSamples; G /= (float) numSamples; B /= (float) numSamples;
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

SColor PathTracer::shadeIntersectionPath(Intersection in, int d) {
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
    Vect Pt = in.calculateIntersectionPoint();

    shade = mat->getAmbColor();

    // Trace shadow scalar towards single light
    std::vector<Light *> lts = scene->getLights();
    if (!lts.empty()) {
        uint p = (uint) ((double) lts.size() * Rand::Random());
        Light *l = lts.at(p);
        float Fattj = calculateFattj(Pt, l);
        if (Fattj > 0) {
            SColor Sj = calculateShadowScalar(l, in, (int) depth);
            shade = shade + directIllumination(l, in, Sj, Fattj);
        }
    }
    SColor diffRefl = diffuseInterreflect(in, d - 1);
    shade = shade + diffRefl;
    
    if (ks.length() > 0) {
        SColor reflection;
        Ray r = in.calculateReflection();
        Intersection rin = scene->intersectsWithBVHTree(r);
        reflection = shadeIntersectionPath(rin, d - 1).linearMult(ks);
        shade = shade + reflection;
    }

    if (kt > 0) {
        SColor refraction;
        Ray r;
        if (in.calculateRefraction(r)) {
            Intersection rin = scene->intersectsWithBVHTree(r);
            refraction = shadeIntersectionPath(rin, d - 1).linearMult(kt);
        } else {
            refraction = SColor(0, 0, 0);
        }
        shade = shade + refraction;
    }

    return shade;
}

SColor PathTracer::diffuseInterreflect(Intersection intersection, int d) {
    Vect norm = intersection.calculateSurfaceNormal();
    Vect rayDir = specularSampleUpperHemisphere(intersection);
    // Vect rayDir = uniformSampleUpperHemisphere(norm);
    Ray diffuseRay(intersection.calculateIntersectionPoint() + norm.linearMult(0.0001f), rayDir);

    SColor albedo = intersection.getColor();

    intersection = scene->intersectsWithBVHTree(diffuseRay);
    if (intersection.hasIntersected()) {
        // SColor diffRefl = intersection.getColor();
        // float cos_theta = rayDir.dotProduct(norm);
        // SColor BRDF = 2 * diffRefl * cos_theta;
        SColor reflected = shadeIntersectionPath(intersection, d);
        // return reflected * BRDF /* + diffRefl * 0.2f */;
        return albedo * reflected;
    }

    if (usingEnvMap) {
        return envMap.getTexel(rayDir);
    } else {
        return SColor();
    }
}

Vect PathTracer::uniformSampleUpperHemisphere(Vect &sampleDir) {
    float x = 1 - 2 * (float) Rand::Random();
    float y = 1 - 2 * (float) Rand::Random();
    float z = 1 - 2 * (float) Rand::Random();

    Vect sample(x, y, z);
    sample.normalize();
    if (sample.dotProduct(sampleDir) < 0)
        return sample.invert();
 return sample;
}

Vect PathTracer::specularSampleUpperHemisphere(Intersection &ins) {
    // PI < x, y, < PI
    float specular = ins.specColor().length();
    float x = (1 - specular) * (M_PI - 2 * M_PI * (float) Rand::Random());
    float y = (1 - specular) * (M_PI - 2 * M_PI * (float) Rand::Random());

    Vect dir = ins.calculateReflection().getDirection();
    Vect_h dir_h(dir.getX(), dir.getY(), dir.getZ());
    Vect_h::Rotate(dir_h, 'x', x);
    Vect_h::Rotate(dir_h, 'y', y);

    Vect sample(dir_h.getX(), dir_h.getY(), dir_h.getZ());
    sample.normalize();
    if (sample.dotProduct(ins.calculateSurfaceNormal()) < 0)
        return sample.invert();
    return sample;
}

bool PathTracer::russianRoulette(SColor refl, float &survivorMult) {
    float p = max(refl.R(), max(refl.G(), refl.B()));
    survivorMult = 1.0f / p;
    if (Rand::Random() > p) 
        return true;
    return false;
}
