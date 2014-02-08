#include "rayscene.h"

Vect Light::getIntensity() {
    Vect i;
    i.setX(color.R / 255.0f);
    i.setY(color.G / 255.0f);
    i.setZ(color.B / 255.0f);
    return i;
}

RayScene::RayScene() {

}

RayScene::~RayScene() {
    for (uint i = 0; i < shapes.size(); i++) {
        Shape *s = shapes.at(i);
        delete s;
    }
}

void RayScene::setLights(std::vector<Light> ls) {
    lights = ls;
}

void RayScene::addLight(Light l) {
    lights.push_back(l);
}

Light RayScene::getLight(uint pos) {
    return lights.at(pos);
}

void RayScene::setShapes(std::vector<Shape *> ss) {
    shapes = ss;
}

void RayScene::addShape(Shape *s) {
    shapes.push_back(s);
}

Shape * RayScene::getShape(uint pos) {
    return shapes.at(pos);
}

Intersection RayScene::calculateRayIntersection(Ray ray) {
    for (uint i = 0; i < shapes.size(); i++) {
        float t;
        Shape *s = shapes.at(i);
        if (s->intersects(ray, t)) {
            Intersection is(ray);
            is.setIntersectionPoint(t);
            return is;
        }
    }
    return Intersection();
}

std::string RayScene::toString() const {
    stringstream s;
    s << "RayScene" << endl;
    s << "\tCamera: " << endl;
    s << "\tLigts:" << endl;
    for (uint i = 0; i < lights.size(); i++) {
        Light l = lights.at(i);
        s << "\t\tL" << i << ": " << l.getPos() << endl;
    }
    s << "\tShapes:" << endl;
    for (uint i = 0; i < shapes.size(); i++) {
        Shape *sp = shapes.at(i);
        ShapeType t = sp->getType();
        s << "\t\tS: " << t << endl;
    }
    return s.str();
}

ostream& operator <<(ostream& stream, const RayScene &scene) {
    stream << scene.toString();
    return stream;
}
