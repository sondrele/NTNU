#include "material.h"

Material::Material() {
    shininess = 1;
    transparency = 0;
}

bool Material::isReflective() {
    return specColor.length() > 0;
}

bool Material::isRefractive() {
    return transparency > 0;
}
