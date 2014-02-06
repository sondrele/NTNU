#include "rayscene.h"

RayScene::RayScene() {

}

RayScene::~RayScene() {
    // for (std::vector<Shape *>::iterator it = shapes.begin(); it != shapes.end(); ++it) {
    //     delete * it;  
    //     it = shapes.erase(it);
    // }
}

void RayScene::setCameraPos(Vect pos) {
    camera.setPos(pos);
}

void RayScene::setCameraViewDir(Vect viewDir) {
    camera.setViewDir(viewDir);
}

void RayScene::setCameraOrthoUp(Vect orthoUp) {
    camera.setOrthoUp(orthoUp);
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
    for (uint i = 0; i < ss.size(); i++) {
        Shape *sp = ss.at(i);
        ShapeType t = sp->getType();
        cout << "\t\tS: " << t << endl;
    }
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
