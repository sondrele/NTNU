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

Sphere * RaySceneFactory::NewSphere(float radius, Vect origin) {
    Sphere *s = new Sphere();
    s->setRadius(radius);
    s->setOrigin(origin);
    return s;
}

void RaySceneFactory::CreateMaterial(Material &m, MaterialIO &mio) {
    m.setDiffColor(SColor(mio.diffColor));
    m.setAmbColor(SColor(mio.ambColor));
    m.setSpecColor(SColor(mio.specColor));
    // m.setEmissColor(SColor(mio.emissColor));
    m.setShininess(mio.shininess);
    m.setTransparency(mio.ktran);
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
    l.setColor(SColor(lio.color));
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

void RaySceneFactory::AddMaterials(Shape *s, MaterialIO *mio, long numMaterials) {
    for (long i = 0; i < numMaterials; i++) {
        Material m;
        RaySceneFactory::CreateMaterial(m, mio[i]);
        s->addMaterial(m);
        // cout << "Adding material to: " << s->getType() << endl;
    }
}

Shape * RaySceneFactory::CreateShape(ObjIO &oio) {
    switch(oio.type) {
        case SPHERE_OBJ: {
            Sphere *s = new Sphere();
            SphereIO sio = *((SphereIO *) oio.data);
            RaySceneFactory::CreateSphere(*s, sio);
            RaySceneFactory::AddMaterials(s, oio.material, oio.numMaterials);
            return s;
        }
        case POLYSET_OBJ: {
            Mesh *m = new Mesh();
            PolySetIO pio = *((PolySetIO *) oio.data);
            RaySceneFactory::CreateMesh(*m, pio);
            RaySceneFactory::AddMaterials(m, oio.material, oio.numMaterials);
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
    std::vector<Light> lts;
    RaySceneFactory::CreateLights(lts, *sio.lights);
    std::vector<Shape *> shps;
    RaySceneFactory::CreateShapes(shps, *sio.objects);

    // Camera cam;
    // RaySceneFactory::CreateCamera(cam, *sio.camera);
    // s.setCamera(cam);
    s.setLights(lts);
    s.setShapes(shps);
}
