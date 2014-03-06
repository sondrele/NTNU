#include "rscenefactory.h"

PX_Color RSceneFactory::ColorToPX_Color(Color c) {
    PX_Color color;
    color.R = (unsigned char) (c[0] * 255);
    color.G = (unsigned char) (c[1] * 255);
    color.B = (unsigned char) (c[2] * 255);
    return color;
}

Vect RSceneFactory::PointToVect(Point p) {
    Vect v;
    v.setX(p[0]);
    v.setY(p[1]);
    v.setZ(p[2]);
    return v;
}

Vertex RSceneFactory::PointToVertex(Point p) {
    Vertex v;
    v.setX(p[0]);
    v.setY(p[1]);
    v.setZ(p[2]);
    return v;
}

Light * RSceneFactory::CreateLight(LightIO &lio) {
    Light *l = new Light();
    l->setType(lio.type);
    l->setIntensity(SColor(lio.color));
    switch(lio.type) {
        case POINT_LIGHT: {
            l->setPos(RSceneFactory::PointToVect(lio.position));
            break;
        }
        case DIRECTIONAL_LIGHT: {
            l->setDir(RSceneFactory::PointToVect(lio.direction));
            break;
        }
        case SPOT_LIGHT:
        default:
        break;
    }
    return l;
}

void RSceneFactory::CreateLights(std::vector<Light *> &lts, LightIO &lio) {
    LightIO *temp = &lio;
    while (temp != NULL) {
        Light *l = RSceneFactory::CreateLight(*temp);
        lts.push_back(l);
        temp = temp->next;
    }
}

Sphere * RSceneFactory::NewSphere(float radius, Vect origin) {
    Sphere *s = new Sphere();
    s->setRadius(radius);
    s->setOrigin(origin);
    return s;
}

void RSceneFactory::CreateSphere(Sphere &sphere, SphereIO &s) {
    sphere.setRadius(s.radius);
    sphere.setOrigin(RSceneFactory::PointToVect(s.origin));
    sphere.setX(s.xlength, RSceneFactory::PointToVect(s.xaxis));
    sphere.setY(s.ylength, RSceneFactory::PointToVect(s.yaxis));
    sphere.setZ(s.zlength, RSceneFactory::PointToVect(s.zaxis));
}

Vertex RSceneFactory::CreateVertex(VertexIO &vio) {
    Vertex v = RSceneFactory::PointToVertex(vio.pos);
    v.setMaterialIndex((uint)vio.materialIndex);
    return v;
}

Vertex RSceneFactory::CreateVertexWithBindings(VertexIO &vio) {
    Vertex v = RSceneFactory::PointToVertex(vio.pos);
    v.setMaterialIndex((uint)vio.materialIndex);
    v.setSurfaceNormal(RSceneFactory::PointToVect(vio.norm));
    v.setTextureCoords(vio.s, vio.t);
    return v;
}

Triangle * RSceneFactory::CreateTriangle(PolygonIO &pio) {
    Triangle *t = new Triangle();
    Vertex v0 = RSceneFactory::CreateVertex(pio.vert[0]);
    Vertex v1 = RSceneFactory::CreateVertex(pio.vert[1]);
    Vertex v2 = RSceneFactory::CreateVertex(pio.vert[2]);
    t->setA(v0);
    t->setB(v1);
    t->setC(v2);
    return t;
}

Triangle * RSceneFactory::CreateTriangleWithBindings(PolygonIO &pio) {
    Triangle *t = new Triangle();
    Vertex v0 = RSceneFactory::CreateVertexWithBindings(pio.vert[0]);
    Vertex v1 = RSceneFactory::CreateVertexWithBindings(pio.vert[1]);
    Vertex v2 = RSceneFactory::CreateVertexWithBindings(pio.vert[2]);
    t->setA(v0);
    t->setB(v1);
    t->setC(v2);
    return t;
}

void RSceneFactory::CreateMesh(Mesh &m, PolySetIO &pio,
    std::vector<Material *> mats) 
{
    for (int i = 0; i < pio.numPolys; i++) {
        Triangle *t;
        if (pio.normType == PER_VERTEX_NORMAL) {
            m.perVertexNormal(true);
            t = RSceneFactory::CreateTriangleWithBindings(pio.poly[i]);
        } else {
            t = RSceneFactory::CreateTriangle(pio.poly[i]);
        }
        
        if (pio.materialBinding == PER_VERTEX_MATERIAL) {
            m.perVertexMaterial(true);
            t->setMaterial(mats[t->getA().getMaterialIndex()], 'a');
            t->setMaterial(mats[t->getB().getMaterialIndex()], 'b');
            t->setMaterial(mats[t->getC().getMaterialIndex()], 'c');
        } else {
            t->setMaterial(mats[t->getA().getMaterialIndex()], 'a');
        }
        m.addTriangle(t);
    }
}

Material * RSceneFactory::CreateMaterial(MaterialIO &mio) {
    Material *m = new Material();
    m->setDiffColor(SColor(mio.diffColor));
    m->setAmbColor(SColor(mio.ambColor));
    m->setSpecColor(SColor(mio.specColor));
    // m->setEmissColor(SColor(mio.emissColor));
    m->setShininess(mio.shininess);
    m->setTransparency(mio.ktran);
    return m;
}

std::vector<Material *> RSceneFactory::CreateMaterials(MaterialIO *mio, long numMaterials) {
    std::vector<Material *> mats;
    for (long i = 0; i < numMaterials; i++) {
        Material *m = RSceneFactory::CreateMaterial(mio[i]);
        mats.push_back(m);
    }
    return mats;
}

void RSceneFactory::AddMaterials(Shape *s, std::vector<Material *> mats) {
    for (uint i = 0; i < mats.size(); i++) {
        s->addMaterial(mats[i]);
    }
}

Shape * RSceneFactory::CreateShape(ObjIO &oio) {
    switch(oio.type) {
        case SPHERE_OBJ: {
            Sphere *s = new Sphere();
            SphereIO sio = *((SphereIO *) oio.data);
            std::vector<Material *> mats = RSceneFactory::CreateMaterials(oio.material, oio.numMaterials);
            RSceneFactory::AddMaterials(s, mats);
            RSceneFactory::CreateSphere(*s, sio);
            return s;
        }
        case POLYSET_OBJ: {
            Mesh *m = new Mesh();
            PolySetIO pio = *((PolySetIO *) oio.data);
            std::vector<Material *> mats = RSceneFactory::CreateMaterials(oio.material, oio.numMaterials);
            RSceneFactory::AddMaterials(m, mats);
            RSceneFactory::CreateMesh(*m, pio, mats);
            return m;
        }
        default:
        return NULL;
    }
}

void RSceneFactory::CreateShapes(std::vector<Shape *> &shps, ObjIO &oio) {
    ObjIO *temp = &oio;
    while (temp != NULL) {
        Shape *s = RSceneFactory::CreateShape(*temp);
        shps.push_back(s);
        temp = temp->next;
    }
}

void RSceneFactory::CreateScene(RScene &s, SceneIO &sio) {
    std::vector<Light *> lts;
    RSceneFactory::CreateLights(lts, *sio.lights);
    std::vector<Shape *> shps;
    RSceneFactory::CreateShapes(shps, *sio.objects);

    s.setLights(lts);
    s.setShapes(shps);
}

void RSceneFactory::CreateCamera(Camera &c, CameraIO &cio) {
    c.setPos(RSceneFactory::PointToVect(cio.position));
    c.setViewDir(RSceneFactory::PointToVect(cio.viewDirection));
    c.setOrthoUp(RSceneFactory::PointToVect(cio.orthoUp));
    c.setFocalDist(cio.focalDistance);
    c.setVerticalFOV(cio.verticalFOV);
}

// Parse obj
Mesh * RSceneFactory::CreateMeshFromObj(tinyobj::mesh_t msh, Material *mat) {
    Vertex a, b, c;
    Mesh *m = new Mesh();
    std::vector<Vertex> vertexes;

    assert(msh.positions.size() % 3 == 0);
    for (uint i = 0; i < msh.positions.size(); i += 3) {
        Vertex v(msh.positions[i], msh.positions[i + 1], msh.positions[i + 2]);
        vertexes.push_back(v);
    }

    assert(msh.indices.size() % 3 == 0);
    for (uint i = 0; i < msh.indices.size(); i += 3) {
        int a = msh.indices[i];
        int b = msh.indices[i + 1];
        int c = msh.indices[i + 2];

        Triangle *t = new Triangle();
        t->setA(vertexes[a]);
        t->setB(vertexes[b]);
        t->setC(vertexes[c]);
        t->setMaterial(mat, 'a');
        m->addTriangle(t);
    }

    // TODO: Add normals

    return m;
}

Material * RSceneFactory::CreateMaterialFromObj(tinyobj::material_t mat) {
    Material *m = new Material();
    m->setDiffColor(SColor(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]));
    m->setAmbColor(SColor(mat.ambient[0], mat.ambient[1], mat.ambient[2]));
    m->setSpecColor(SColor(mat.specular[0], mat.specular[1], mat.specular[2]));
    // m->setEmissColor(SColor(mat.emission[0], mat.emission[1], mat.emission[2]));
    m->setShininess(mat.shininess);
    m->setTransparency(mat.transmittance[0]);
    return m;
}

Shape * RSceneFactory::CreateShapeFromObj(tinyobj::shape_t shp) {
    Material *mat = RSceneFactory::CreateMaterialFromObj(shp.material);
    Mesh *mesh = RSceneFactory::CreateMeshFromObj(shp.mesh, mat);
    mesh->addMaterial(mat);
    return mesh;
}

void RSceneFactory::CreateShapesFromObj(std::vector<Shape *> &shps,
    std::vector<tinyobj::shape_t> &objshps)
{
    for (uint i = 0; i < objshps.size(); i++) {
        Shape *s = CreateShapeFromObj(objshps[i]);
        shps.push_back(s);
    }
}

void RSceneFactory::CreateSceneFromObj(RScene *scene, std::vector<tinyobj::shape_t> &objshps) {
    std::vector<Shape *> shps;
    RSceneFactory::CreateShapesFromObj(shps, objshps);
    scene->setShapes(shps);
}
