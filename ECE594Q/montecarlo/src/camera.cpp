#include "camera.h"

Camera::Camera() {
    
}

void Camera::setPos(Vect p) {
    pos = p;
}

void Camera::setViewDir(Vect d) {
    viewDir = d;
    viewDir.normalize();
}

void Camera::setOrthoUp(Vect d) {
    orthoUp = d;
    orthoUp.normalize();
}
