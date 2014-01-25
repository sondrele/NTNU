#include "FrameBuffer.h"

#define FX(x, y) (y * WIDTH + x)
#define PX(x, y) (y * m + x)

FrameBuffer::FrameBuffer(uint width, uint height, uint m, uint n,
    float hither, float yon, float fov, float aspectRatio) 
{
    WIDTH = width;
    HEIGHT = height;
    this->m = m;
    this->n = n;
    this->hither = hither;
    this->yon = yon;
    this->fov = fov;
    this->aspectRatio = aspectRatio;

    for (uint y = 0; y < height; y++) {
        for (uint x = 0; x < width; x++) {
            pixels.push_back(FramePixel(x, y, m, n));
        }
    }
}

FrameBuffer::FrameBuffer(uint width, uint height)
: FrameBuffer(width, height, 1, 1, 1, 100, M_PI / 3.0, 1) {;}

FrameBuffer::FrameBuffer(uint width, uint height, uint m, uint n)
: FrameBuffer(width, height, m, n, 1, 100, M_PI / 3.0, 1) {;}

void FrameBuffer::setPixel(uint x, uint y, uint m, uint n, PX_Color color) {
    if (m < this->m && n < this->n && x < this->WIDTH && y < this->HEIGHT) {
        pixels[FX(x, y)].setSample(m, n, color);
    } else
        throw "Cannot set sample on pixel";
}

FramePixel FrameBuffer::getPixel(uint x, uint y) {
    if (x < this->WIDTH && y < this->HEIGHT)
        return pixels[FX(x, y)];
    else
        throw "Cannot get pixel";
}

void FrameBuffer::addPoint(MeshPoint m) {
    points.push_back(m);
}

MeshPoint FrameBuffer::getPoint(uint index) {
    return points[index];
}

void FrameBuffer::addMicroPolygon(MicroPolygon &mp) {
    mPolygons.push_back(mp);
}

void FrameBuffer::addMesh(Mesh &object) {
    objects.push_back(&object);
    for (uint i = 0; i < object.getSize(); i++) {
        addPoint(object.getPoint(i));
    }
}

bool FrameBuffer::projectAndScalePoint(Vect &vect) {
    Vect::Project(vect, hither, yon, fov, aspectRatio);
    vect.homogenize();
    if (abs(vect.getX()) <= 1.0 and abs(vect.getY()) <= 1) {
        vect.setX(vect.getX() + 1);
        vect.setY(vect.getY() + 1);

        Vect::Scale(vect, WIDTH / 2.0, HEIGHT / 2.0, 1);
        return true;
    } else return false;
}

bool FrameBuffer::projectMicroPolygon(MicroPolygon &mp) {
    bool b;
    b = projectAndScalePoint(mp.a);
    b &= projectAndScalePoint(mp.b);
    b &= projectAndScalePoint(mp.c);
    b &= projectAndScalePoint(mp.d);
    return b;
}

void FrameBuffer::plotPoints(const char *name) {
    cimg_library::CImg<unsigned char> image(WIDTH, HEIGHT, 1, 3, 0);

    const unsigned char white[] = {255, 255, 255};
    // Project each point of matrix into screen space
    for (uint i = 0; i < points.size(); i++) {
        Vect point = points.at(i);
        projectAndScalePoint(point);
        image.draw_point(point.getX(), point.getY(), white);
    }

    image.save_jpeg(name, 100);
}

void FrameBuffer::drawShapes(const char *name) {
    // Project each of the micropolygons to screen space
    std::vector<MicroPolygon> projectedMicroPolygons;
    for (uint i = 0; i < objects.size(); ++i) {
        Mesh *object = objects.at(i);
        std::vector<MicroPolygon> polygons = object->getMicroPolygons();

        for (uint j = 0; j < polygons.size(); j++) {
            MicroPolygon poly = polygons.at(j);
            bool projected = projectMicroPolygon(poly);
            if (projected)
                projectedMicroPolygons.push_back(poly);
        }
    }
    // For each micropolygon, check what samples it intersects
    for (uint i = 0; i < projectedMicroPolygons.size(); i++) {
        MicroPolygon poly = projectedMicroPolygons.at(i);
        
        BoundingBox box = poly.getBoundingBox();
        // if (true || box.X_start >= 0 and box.X_stop <= WIDTH and
        //     box.Y_start >= 0 and box.Y_stop <= HEIGHT) 
        // {
            Vect point(0, 0, 0);
            for (int y = box.Y_start; y < box.Y_stop; y++) {
                for (int x = box.X_start; x < box.X_stop; x++) {
                    unsigned char *mpColor = poly.getColor();
                    // int sampledColor[3] = {0, 0, 0};
                    // Loop over samples to get the right color
                    for (uint dn = 0; dn < n; dn++) {
                        for (uint dm = 0; dm < m; dm++) {
                            float dx = dm / (float)m;
                            float dy = dn / (float)n;
                            point.setX(x + dx);
                            point.setY(y + dy);
                            if (poly.intersects(point)) {
                                pixels[FX(x, y)].setSample(dm, dn, mpColor);
                            }
                        }
                    }
                }
            // }
        }
    }
    exportImage(name);
}

void FrameBuffer::exportImage(const char *name) {
    cimg_library::CImg<unsigned char> image(WIDTH, HEIGHT, 1, 3, 0);

    for (uint i = 0; i < pixels.size(); i++) {
        FramePixel px = pixels[i];
        PX_Color c0 = px.getColor();
        unsigned char color[3] = {
            (unsigned char) c0.R,
            (unsigned char) c0.G,
            (unsigned char) c0.B
        };
        image.draw_point(px.getX(), px.getY(), color);
    }

    image.save_jpeg(name, 100);
}

//*********************************************************************

FramePixel::FramePixel(uint x, uint y, uint m, uint n) {
    this->x = x;
    this->y = y;
    this->m = m;
    this->n = n;
    samples = std::vector<PX_Color>(m * n, {0, 0, 0});
}

void FramePixel::setSample(uint x, uint y, PX_Color color) {
    samples[PX(x, y)] = color;
}

void FramePixel::setSample(uint x, uint y, unsigned char *c) {
    samples[PX(x, y)] = {c[0], c[1], c[2]};
}

PX_Color FramePixel::getColor() {
    uint R = 0, G = 0, B = 0;
    for (uint i = 0; i < samples.size(); i++) {
        R += samples[i].R;
        G += samples[i].G;
        B += samples[i].B;
    }
    R /= m * n;
    G /= m * n;
    B /= m * n;
    return {(unsigned char) R, (unsigned char) G, (unsigned char) B};
}