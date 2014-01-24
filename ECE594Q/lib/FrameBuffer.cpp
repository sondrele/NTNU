#include "FrameBuffer.h"

FrameBuffer::FrameBuffer(uint width, uint height) {
    WIDTH = width;
    HEIGHT = height;
    hither = 1;
    yon = 100;
    fov = M_PI / 3.0;
    aspectRatio = 1;
    m = n = 1;

    for (uint i = 0; i < height; i++) {
        for (uint j = 0; j < width; j++) {
            pixels.push_back(FramePixel(j, i));
        }
    }
}

FrameBuffer::FrameBuffer(uint width, uint height, float hither, float yon,
    float fov, float aspectRatio) 
{
    WIDTH = width;
    HEIGHT = height;
    this->hither = hither;
    this->yon = yon;
    this->fov = fov;
    this->aspectRatio = aspectRatio;
    m = n = 1;

    for (uint i = 0; i < height; i++) {
        for (uint j = 0; j < width; j++) {
            pixels.push_back(FramePixel(j, i));
        }
    }
}

FrameBuffer::FrameBuffer(uint width, uint height, uint m, uint n) {
    WIDTH = width;
    HEIGHT = height;
    this->m = m;
    this->n = n;

    for (uint i = 0; i < height; i++) {
        for (uint j = 0; j < width; j++) {
            pixels.push_back(FramePixel(j, i));
        }
    }
}

void FrameBuffer::addPoint(Vect m) {
    points.push_back(m);
}

Matrix FrameBuffer::getPoint(uint index) {
    return points.at(index);
}

void FrameBuffer::addMesh(Mesh &object) {
    objects.push_back(&object);
    for (uint i = 0; i < object.getSize(); i++) {
        Vect point = object.getPoint(i).point;
        addPoint(point);
    }
}

void FrameBuffer::projectAndScalePoint(Vect &vect) {
    Vect::Project(vect, hither, yon, fov, aspectRatio);
    vect.homogenize();
    vect.setX(vect.getX() + 1);
    vect.setY(vect.getY() + 1);

    Vect::Scale(vect, WIDTH / 2.0, HEIGHT / 2.0, 1);
}

void FrameBuffer::projectMeshPoint(MeshPoint &mp) {
    projectAndScalePoint(mp.point);
}

void FrameBuffer::projectMicroPolygon(MicroPolygon &mp) {
    projectMeshPoint(mp.a);
    projectMeshPoint(mp.b);
    projectMeshPoint(mp.c);
    projectMeshPoint(mp.d);
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

void FrameBuffer::drawMicroPolygons(const char *name) {
    cimg_library::CImg<unsigned char> image(WIDTH, HEIGHT, 1, 3, 0);

    // const unsigned char white[] = {255, 255, 255};
    // Project each of the micropolygons to screen space
    std::vector<MicroPolygon> projectedMicroPolygons;
    for (uint i = 0; i < objects.size(); ++i) {
        Mesh *object = objects.at(i);
        std::vector<MicroPolygon> polygons = object->getMicroPolygons();

        for (uint j = 0; j < polygons.size(); j++) {
            MicroPolygon poly = polygons.at(j);
            projectMicroPolygon(poly);
            projectedMicroPolygons.push_back(poly);
        }
    }
    // For each micropolygon, check what samples it intersects
    for (uint i = 0; i < projectedMicroPolygons.size(); i++) {
        MicroPolygon poly = projectedMicroPolygons.at(i);
        
        float *f = poly.getBoundingBox();
        int px0_x = (int) (floor(f[0]));
        int px0_y = (int) (floor(f[1]));
        int px1_x = (int) (ceil(f[2]));
        int px1_y = (int) (ceil(f[3]));

        Vect point(0, 0, 0);
        for (int y = px0_y; y < px1_y; y++) {
            for (int x = px0_x; x < px1_x; x++) {
                unsigned char *mpColor = poly.getColor();
                // int sampledColor[3] = {0, 0, 0};
                // Loop over samples to get the right color
                for (uint m = 0; m < this->m; m++) {
                    for (uint n = 0; n < this->n; n++) {
                        float dx = m / (float)this->m;
                        float dy = n / (float)this->n;
                        point.setX(x + dx);
                        point.setY(y + dy);
                        if (poly.intersects(point)) {
                            pixels[y * WIDTH + x].addSample(mpColor);
                        }
                    }
                }
            }
        }
    }

    for (uint i = 0; i < pixels.size(); i++) {
        FramePixel px = pixels[i];
        F_Color c0 = px.getColor();
        unsigned char c[] = {c0.R, c0.G, c0.B};
        image.draw_point(px.getX(), px.getY(), c);
    }

    image.save_jpeg(name, 100);
}

FramePixel::FramePixel(uint X, uint Y) {
    this->X = X;
    this->Y = Y;
}

void FramePixel::addSample(F_Color color) {
    samples.push_back(color);
}

void FramePixel::addSample(unsigned char *color) {
    samples.push_back({color[0], color[1], color[2]});
}

F_Color FramePixel::getColor() {
    if (samples.size() > 0) {
        F_Color c = {0, 0, 0};
        for (uint i = 0; i < samples.size(); i++) {
            c.R += samples.at(i).R;
            c.G += samples.at(i).G;
            c.B += samples.at(i).B;
        }
        c.R /= samples.size();
        c.G /= samples.size();
        c.B /= samples.size();
        return c;
    } else {
        return {0, 0, 0};
    }
}