#include "rscene.h"

void Light::setArea(Vect min, Vect max) {
    type = AREA_LIGHT;
    area.setMin(min);
    area.setMin(max);
}

bool Light::intersects(Ray ray) {
    if (type == AREA_LIGHT) {
        return area.intersects(ray);
    }
    return false;
}

RScene::RScene() {

}

RScene::~RScene() {
    for (uint i = 0; i < shapes.size(); i++) {
        Shape *s = shapes.at(i);
        delete s;
    }

    for (uint i = 0; i < lights.size(); i++) {
        delete lights.at(i);
    }
}

void RScene::setLights(std::vector<Light *> ls) {
    lights = ls;
}

void RScene::addLight(Light *l) {
    lights.push_back(l);

}

Light * RScene::getLight(uint pos) {
    return lights.at(pos);
}

void RScene::setShapes(std::vector<Shape *> ss) {
    shapes = ss;
    std::vector<Shape *> shps;
    for (uint i = 0; i < shapes.size(); i++) {
        Shape *s0 = shapes[i];
        if (s0->getType() == MESH) {
            std::vector<Triangle *> ts = ((Mesh *) s0)->getTriangles();
            shps.insert(shps.end(), ts.begin(), ts.end());
        } else
            shps.push_back(s0);
    }
    cout << "Number of shapes: " << shps.size() << endl;
    searchTree.buildTree(shps);
}

void RScene::addShape(Shape *s) {
    shapes.push_back(s);
    searchTree.buildTree(shapes);
}

Shape * RScene::getShape(uint pos) {
    return shapes.at(pos);
}

Intersection RScene::intersects(Ray ray) {
    return searchTree.intersects(ray);
}
