#include "rayscene_factory.h"

PX_Color RaySceneFactory::ColorToPX_Color(Color c) {
    PX_Color color;
    color.R = (unsigned char) (c[0] * 255);
    color.G = (unsigned char) (c[1] * 255);
    color.B = (unsigned char) (c[2] * 255);
    return color;
}

Vect RaySceneFactory::PointToVect(Point p) {
    Vect v;
    v.setX(p[0]);
    v.setY(p[1]);
    v.setZ(p[2]);
    return v;
}

void RaySceneFactory::CreateSphere(Sphere &sphere, SphereIO &s) {
    sphere.setRadius(s.radius);
    sphere.setOrigin(RaySceneFactory::PointToVect(s.origin));
    sphere.setX(s.xlength, RaySceneFactory::PointToVect(s.xaxis));
    sphere.setY(s.ylength, RaySceneFactory::PointToVect(s.yaxis));
    sphere.setZ(s.zlength, RaySceneFactory::PointToVect(s.zaxis));
}

void RaySceneFactory::CreateTriangle(Triangle &t, PolygonIO &pio) {
    if (pio.numVertices == 3) {
        VertexIO p0 = pio.vert[0];
        VertexIO p1 = pio.vert[1];
        VertexIO p2 = pio.vert[2];

        t.setA(RaySceneFactory::PointToVect(p0.pos));
        t.setB(RaySceneFactory::PointToVect(p1.pos));
        t.setC(RaySceneFactory::PointToVect(p2.pos));
    }

}

void RaySceneFactory::CreateMesh(Mesh &m, PolySetIO &pio) {
    for (int i = 0; i < pio.numPolys; i++) {
        Triangle t;
        RaySceneFactory::CreateTriangle(t, pio.poly[i]);
        m.addTriangle(t);
    }
}

void RaySceneFactory::CreateLight(Light &l, LightIO &lio) {
    l.setType(lio.type);
    switch(lio.type) {
        case POINT_LIGHT: {
            l.setPos(RaySceneFactory::PointToVect(lio.position));
            break;
        }
        case DIRECTIONAL_LIGHT: {
            l.setDir(RaySceneFactory::PointToVect(lio.direction));
            break;
        }
        case SPOT_LIGHT:
        default:
        break;
    }
}

void RaySceneFactory::CreateLights(std::vector<Light> &lts, LightIO &lio) {
    LightIO *temp = &lio;
    while (temp != NULL) {
        Light l;
        RaySceneFactory::CreateLight(l, *temp);
        lts.push_back(l);
        temp = temp->next;
    }
}

void RaySceneFactory::CreateCamera(Camera &c, CameraIO &cio) {
    c.setPos(RaySceneFactory::PointToVect(cio.position));
    c.setViewDir(RaySceneFactory::PointToVect(cio.viewDirection));
    c.setOrthoUp(RaySceneFactory::PointToVect(cio.orthoUp));
    c.setFocalDist(cio.focalDistance);
    c.setVerticalFOV(cio.verticalFOV);
}
