#include "Mesh.h"

#define PT(i, j) (i * width + j)

MeshPoint::MeshPoint(Vect &v) : Vect(v) {;}

MeshPoint::MeshPoint(float x, float y, float z) : Vect(x, y, z) {
    setW(1);
}

MicroPolygon::MicroPolygon() {
    opacity = 1.0;
    color[0] = 255;
    color[1] = 255;
    color[2] = 255;
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

bool MicroPolygon::intersects(Vect point) {
    bool intersectsT1 = Utils::PointInTriangle(point, a, b, c);
    bool intersectsT2 = Utils::PointInTriangle(point, b, d, c);
    return intersectsT1 or intersectsT2;
}

BoundingBox MicroPolygon::getBoundingBox() {
    BoundingBox bound;
    float x_min = min(a.getX(), min(b.getX(), min(c.getX(), d.getX())));
    float y_min = min(a.getY(), min(b.getY(), min(c.getY(), d.getY())));
    float x_max = max(a.getX(), max(b.getX(), max(c.getX(), d.getX())));
    float y_max = max(a.getY(), max(b.getY(), max(c.getY(), d.getY())));

    bound.X_start = (uint) floor(x_min);
    bound.Y_start = (uint) floor(y_min);
    bound.X_stop = (uint) ceil(x_max);
    bound.Y_stop = (uint) ceil(y_max);
    return bound;
}

float MicroPolygon::getDepth() {
    return (a.getZ() + b.getZ() + c.getZ() + d.getZ()) / 4.0;
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
        return points[index];
    }
    throw "MeshPointException: Cannot get point";
}

std::string Mesh::toString() {
    stringstream s;
    s << "Mesh[" << getSize() << "]\n";
    return s.str();
}

std::vector<MicroPolygon> Mesh::getMicroPolygons() {
    std::vector<MicroPolygon> mPolygons;
    uint x = 255;
    bool decrementColor = false;
    for (uint i = 0; i < getSize()-getWidth(); i++) {
        MicroPolygon mp;
        if ((i + 1) % getWidth() == 0) {
            decrementColor = true;
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
        if (decrementColor) {
            decrementColor = false;
            x -= (255 / getWidth());
        }
        // mp.setOpacity(1);
        // mp.setColor(0, x, 255);
        mPolygons.push_back(mp);
    }
    return mPolygons;
}

void Mesh::rotate(const char axis, const float degrees) {
    double radians = degrees * U_PI / 180.0;
    for (uint i = 0; i < points.size(); i++) {
        Vect::Rotate(points[i], axis, radians);
    }
}

void Mesh::translate(const float dx, const float dy, const float dz) {
    for (uint i = 0; i < points.size(); i++) {
        Vect::Translate(points[i], dx, dy, dz);
    }   
}

RiSphere::RiSphere(float radius, uint resolution):
Mesh(resolution, resolution) 
{
    this->radius = radius;
    this->resolution = resolution;

    double lat, lon;
    for (uint i = 0; i < resolution; i++) {
        lat = (2 * U_PI) * i / resolution;
        for (uint j = 0; j < resolution; j++) {
            lon = (2 * U_PI) * j / resolution;
            float x = radius * cos(lon) * sin(lat);
            float y = radius * cos(lat);
            float z = radius * sin(lon) * sin(lat);
            // TODO: Fiks z!
            MeshPoint p(x, y, z);
            addPoint(p);
        }
    }
}

RiRectangle::RiRectangle(float width, float height,
    float depth) : Mesh(4, 2) {
    this->width = width;
    this->height = height;
    this->depth = depth;

    addPoint(MeshPoint(-width/2.0, -height/2.0, depth));
    addPoint(MeshPoint(-width/2.0, -height/2.0, depth+depth));
    addPoint(MeshPoint(-width/2.0,  height/2.0, depth));
    addPoint(MeshPoint(-width/2.0,  height/2.0, depth+depth));
    addPoint(MeshPoint(width/2.0,  -height/2.0, depth));
    addPoint(MeshPoint(width/2.0,  -height/2.0, depth+depth));
    addPoint(MeshPoint(width/2.0,   height/2.0, depth));
    addPoint(MeshPoint(width/2.0,   height/2.0, depth+depth));
}
