#include "Mesh.h"

MeshPoint::MeshPoint(float x, float y, float z) {
    point.setX(x);
    point.setY(y);
    point.setZ(z);
    point.setW(1);
    color[0] = 0;
    color[1] = 0;
    color[2] = 0;
}

void MicroPolygon::setColor(unsigned char *RGB) {
    color[0] = RGB[0];
    color[1] = RGB[1];
    color[2] = RGB[2];
}

void MicroPolygon::setColor(unsigned char R, unsigned char G, 
    unsigned char B) 
{
    color[0] = R;
    color[1] = G;
    color[2] = B;
}

// bool MicroPolygon::isNeighbour(MicroPolygon other) {
//     bool r = this->b == other.a and this->d == other.c;
//     bool l = this->a == other.b and this->c == other.d;
//     bool u = this->a == other.c and this->b == other.d;
//     bool d = this->c == other.a and this->d == other.b;
//     return r or l or u or d;
// }

bool MicroPolygon::intersects(Vect point) {
    bool intersectsT1 = Utils::PointInTriangle(point, a.point, b.point, c.point);
    bool intersectsT2 = Utils::PointInTriangle(point, b.point, d.point, c.point);
    return intersectsT1 or intersectsT2;
}

float* MicroPolygon::getBoundingBox() {
    float *bound = new float[4];
    float x_min = min(a.getX(), min(b.getX(), min(c.getX(), d.getX())));
    float y_min = min(a.getY(), min(b.getY(), min(c.getY(), d.getY())));
    float x_max = max(a.getX(), max(b.getX(), max(c.getX(), d.getX())));
    float y_max = max(a.getY(), max(b.getY(), max(c.getY(), d.getY())));

    bound[0] = x_min;
    bound[1] = y_min;
    bound[2] = x_max;
    bound[3] = y_max;
    return bound;
}

Mesh::Mesh(uint width, uint height) {
    this->width = width;
    this->height = height;
}

Mesh::~Mesh() {

}

uint Mesh::getSize() {
    // or width * height
    return points.size();
}

void Mesh::addPoint(MeshPoint point) {
    points.push_back(point);
    if (points.size() > width * height) {
        cout << "size overflow" << endl;
    }
}

MeshPoint Mesh::getPoint(uint index) {
    if (index < points.size()) {
        return points.at(index);
    }
    throw "MeshPointException: Cannot get point";
}

void Mesh::movePoint(uint index, float x, float y, float z) {
    points[index].setX(x);
    points[index].setY(y);
    points[index].setZ(z);
    points[index].setW(1);
}

std::string Mesh::toString() {
    stringstream s;
    s << "Mesh[" << getSize() << "]\n";
    return s.str();
}

std::vector<MicroPolygon> Mesh::getMicroPolygons() {
    std::vector<MicroPolygon> mPolygons;
    for (uint i = 0; i < getSize()-getWidth(); i++) {
        MicroPolygon mp;
        if ((i + 1) % getWidth() == 0) {
            mp.a = points[i];
            mp.b = points[i-getWidth()+1];
            mp.c = points[i+getWidth()];
            mp.d = points[i+1];
        } else {
            mp.a = points[i];
            mp.b = points[i+1];
            mp.c = points[i+getWidth()];
            mp.d = points[i+getWidth()+1];
        }
        mp.setColor(255 * i / getSize(), 255 * i / getSize(), 0);
        mPolygons.push_back(mp);
    }
    return mPolygons;
}

RiSphere::RiSphere(float radius, uint resolution):
Mesh(resolution, resolution) 
{
    this->radius = radius;
    this->resolution = resolution;

    double lat, lon;
    for (uint i = 0; i < resolution; i++) {
        lat = (2 * M_PI) * i / resolution;
        for (uint j = 0; j < resolution; j++) {
            lon = (2 * M_PI) * j / resolution;
            float x = radius * cos(lon) * sin(lat);
            float y = radius * cos(lat);
            float z = radius * sin(lon) * sin(lat);
            // TODO: Fiks z!
            MeshPoint p(x, y, z+50);
            addPoint(p);
        }
    }
}

RiRectangle::RiRectangle(float width, float height,
    float deapth) : Mesh(4, 2) {
    this->width = width;
    this->height = height;
    this->deapth = deapth;

    addPoint(MeshPoint(-width/2.0, -height/2.0, deapth));
    addPoint(MeshPoint(-width/2.0, -height/2.0, deapth+deapth));
    addPoint(MeshPoint(-width/2.0,  height/2.0, deapth));
    addPoint(MeshPoint(-width/2.0,  height/2.0, deapth+deapth));
    addPoint(MeshPoint(width/2.0,  -height/2.0, deapth));
    addPoint(MeshPoint(width/2.0,  -height/2.0, deapth+deapth));
    addPoint(MeshPoint(width/2.0,   height/2.0, deapth));
    addPoint(MeshPoint(width/2.0,   height/2.0, deapth+deapth));
}
