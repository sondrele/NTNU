#include "ray.h"

Ray::Ray() {
    
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
