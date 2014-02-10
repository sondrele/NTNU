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

Vertex RaySceneFactory::PointToVertex(Point p) {
    Vertex v;
    v.setX(p[0]);
    v.setY(p[1]);
    v.setZ(p[2]);
    return v;
}

Light * RaySceneFactory::CreateLight(LightIO &lio) {
    Light *l = new Light();
    l->setType(lio.type);
    l->setIntensity(SColor(lio.color));
    switch(lio.type) {
        case POINT_LIGHT: {
            l->setPos(RaySceneFactory::PointToVect(lio.position));
            break;
        }
        case DIRECTIONAL_LIGHT: {
            l->setDir(RaySceneFactory::PointToVect(lio.direction));
            break;
        }
        case SPOT_LIGHT:
        default:
        break;
    }
    return l;
}

void RaySceneFactory::CreateLights(std::vector<Light *> &lts, LightIO &lio) {
    LightIO *temp = &lio;
    while (temp != NULL) {
        Light *l = RaySceneFactory::CreateLight(*temp);
        lts.push_back(l);
        temp = temp->next;
    }
}

Sphere * RaySceneFactory::NewSphere(float radius, Vect origin) {
    Sphere *s = new Sphere();
    s->setRadius(radius);
    s->setOrigin(origin);
    return s;
}

void RaySceneFactory::CreateSphere(Sphere &sphere, SphereIO &s) {
    sphere.setRadius(s.radius);
    sphere.setOrigin(RaySceneFactory::PointToVect(s.origin));
    sphere.setX(s.xlength, RaySceneFactory::PointToVect(s.xaxis));
    sphere.setY(s.ylength, RaySceneFactory::PointToVect(s.yaxis));
    sphere.setZ(s.zlength, RaySceneFactory::PointToVect(s.zaxis));
}

Vertex RaySceneFactory::CreateVertex(VertexIO &vio) {
    Vertex v = RaySceneFactory::PointToVertex(vio.pos);
    v.setMaterialIndex((uint)vio.materialIndex);
    return v;
}

Triangle * RaySceneFactory::CreateTriangle(PolygonIO &pio) {
    Triangle *t = new Triangle();
    Vertex v0 = RaySceneFactory::CreateVertex(pio.vert[0]);
    Vertex v1 = RaySceneFactory::CreateVertex(pio.vert[1]);
    Vertex v2 = RaySceneFactory::CreateVertex(pio.vert[2]);
    t->setA(v0);
    t->setB(v1);
    t->setC(v2);
    return t;
}

void RaySceneFactory::CreateMesh(Mesh &m, PolySetIO &pio, std::vector<Material *> mats) {
    for (int i = 0; i < pio.numPolys; i++) {
        Triangle *t = RaySceneFactory::CreateTriangle(pio.poly[i]);
        Material *y = new Material();
        *y = *(mats[t->getA().getMaterialIndex()]);
        t->addMaterial(y);
        m.addTriangle(t);
    }
}

Material * RaySceneFactory::CreateMaterial(MaterialIO &mio) {
    Material *m = new Material();
    m->setDiffColor(SColor(mio.diffColor));
    m->setAmbColor(SColor(mio.ambColor));
    m->setSpecColor(SColor(mio.specColor));
    // m->setEmissColor(SColor(mio.emissColor));
    m->setShininess(mio.shininess);
    m->setTransparency(mio.ktran);
    return m;
}

std::vector<Material *> RaySceneFactory::CreateMaterials(MaterialIO *mio, long numMaterials) {
    std::vector<Material *> mats;
    for (long i = 0; i < numMaterials; i++) {
        Material *m = RaySceneFactory::CreateMaterial(mio[i]);
        mats.push_back(m);
    }
    return mats;
}

void RaySceneFactory::AddMaterials(Shape *s, std::vector<Material *> mats) {
    for (uint i = 0; i < mats.size(); i++) {
        s->addMaterial(mats[i]);
    }
}

Shape * RaySceneFactory::CreateShape(ObjIO &oio) {
    switch(oio.type) {
        case SPHERE_OBJ: {
            Sphere *s = new Sphere();
            SphereIO sio = *((SphereIO *) oio.data);
            std::vector<Material *> mats = RaySceneFactory::CreateMaterials(oio.material, oio.numMaterials);
            RaySceneFactory::AddMaterials(s, mats);
            RaySceneFactory::CreateSphere(*s, sio);
            return s;
        }
        case POLYSET_OBJ: {
            Mesh *m = new Mesh();
            PolySetIO pio = *((PolySetIO *) oio.data);
            std::vector<Material *> mats = RaySceneFactory::CreateMaterials(oio.material, oio.numMaterials);
            RaySceneFactory::AddMaterials(m, mats);
            RaySceneFactory::CreateMesh(*m, pio, mats);
            return m;
        }
        default:
        return NULL;
    }
}

void RaySceneFactory::CreateShapes(std::vector<Shape *> &shps, ObjIO &oio) {
    ObjIO *temp = &oio;
    while (temp != NULL) {
        Shape *s = RaySceneFactory::CreateShape(*temp);
        shps.push_back(s);
        temp = temp->next;
    }
}

void RaySceneFactory::CreateScene(RayScene &s, SceneIO &sio) {
    std::vector<Light *> lts;
    RaySceneFactory::CreateLights(lts, *sio.lights);
    std::vector<Shape *> shps;
    RaySceneFactory::CreateShapes(shps, *sio.objects);

    s.setLights(lts);
    s.setShapes(shps);
}

void RaySceneFactory::CreateCamera(Camera &c, CameraIO &cio) {
    c.setPos(RaySceneFactory::PointToVect(cio.position));
    c.setViewDir(RaySceneFactory::PointToVect(cio.viewDirection));
    c.setOrthoUp(RaySceneFactory::PointToVect(cio.orthoUp));
    c.setFocalDist(cio.focalDistance);
    c.setVerticalFOV(cio.verticalFOV);
}
