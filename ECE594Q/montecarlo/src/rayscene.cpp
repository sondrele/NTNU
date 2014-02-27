#include "rayscene.h"

RayScene::RayScene() {
}

RayScene::~RayScene() {
    for (uint i = 0; i < shapes.size(); i++) {
        Shape *s = shapes.at(i);
        delete s;
    }

    for (uint i = 0; i < lights.size(); i++) {
        delete lights.at(i);
    }
}

void RayScene::setLights(std::vector<Light *> ls) {
    lights = ls;
}

void RayScene::addLight(Light *l) {
    lights.push_back(l);

}

Light * RayScene::getLight(uint pos) {
    return lights.at(pos);
}

void RayScene::setShapes(std::vector<Shape *> ss) {
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
    cout << shps.size() << endl;
    searchTree.buildTree(shps);
}

void RayScene::addShape(Shape *s) {
    shapes.push_back(s);
    searchTree.buildTree(shapes);
}

Shape * RayScene::getShape(uint pos) {
    return shapes.at(pos);
}

Intersection RayScene::calculateRayIntersection(Ray ray) {
    Intersection ins;
    for (uint i = 0; i < shapes.size(); i++) {
        Shape *s = shapes.at(i);
        BBox bbox = s->getBBox();
        if (bbox.intersects(ray)) {
            Intersection j = s->intersects(ray);
            if (j.hasIntersected()) {
                if (!ins.hasIntersected()) {
                    ins = j;
                } 
                else if (ins.hasIntersected() &&
                    j.getIntersectionPoint() < ins.getIntersectionPoint())
                {
                    ins = j;
                }
            }
        }
    }
    return ins;
}

Intersection RayScene::intersectsWithBVHTree(Ray ray) {
    return searchTree.intersects(ray);
}
