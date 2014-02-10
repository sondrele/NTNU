#include "ray.h"

Ray::Ray() {
    vacuum = true;
}

Ray::Ray(Vect o, Vect d) {
    origin = o;
    direction = d;
    direction.normalize();
}

void Ray::setDirection(Vect d) {
    direction = d;
    direction.normalize();
}

void Ray::switchMedium() {
    vacuum = !vacuum;
}

bool Ray::inVacuum() {
    return vacuum;
}
