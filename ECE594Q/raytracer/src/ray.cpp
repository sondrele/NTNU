#include "ray.h"

Ray::Ray() {
    
}

Ray::Ray(Vect o, Vect d) {
    origin = o;
    direction = d;
}
