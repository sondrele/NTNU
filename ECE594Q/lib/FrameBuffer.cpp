#include "FrameBuffer.h"

FrameBuffer::FrameBuffer(uint width, uint height) {
    WIDTH = width;
    HEIGHT = height;
    hither = 1;
    yon = 100;
    fov = M_PI / 3.0;
    aspectRatio = 1;
    m = n = 1;
    initSamples();
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
    initSamples();
}

FrameBuffer::FrameBuffer(uint width, uint height, uint m, uint n) {
    WIDTH = width;
    HEIGHT = height;
    this->m = m;
    this->n = n;
    initSamples();
}

void FrameBuffer::initSamples() {
    samples = new float[m * n * 2];
    float dx = 1 / (float) (m + 1);
    float dy = 1 / (float) (n + 1);

    for (uint i = 0; i < m; i++) {
        for (uint j = 0, k = 0; j < n * 2; j += 2, k++) {
            samples[2 * n * i + j] = dx * (1 + i);
            samples[2 * n * i + j + 1] = dy * (1 + k);
        }
    }

}

FrameBuffer::~FrameBuffer() {
    delete [] samples;
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

MeshPoint *FrameBuffer::projectMeshPoint(MeshPoint &mp) {
    projectAndScalePoint(mp.point);
    MeshPoint *projected = new MeshPoint(mp.point.getX(), mp.point.getY(), mp.point.getZ());
    return projected;
}

MicroPolygon *FrameBuffer::projectMicroPolygon(MicroPolygon &mp) {
    MicroPolygon *p = new MicroPolygon();
    p->a = projectMeshPoint(*mp.a);
    p->b = projectMeshPoint(*mp.b);
    p->c = projectMeshPoint(*mp.c);
    p->d = projectMeshPoint(*mp.d);
    p->setColor(mp.getColor());
    return p;
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
    std::vector<MicroPolygon *> projectedMicroPolygons;
    for (uint i = 0; i < objects.size(); ++i) {
        std::vector<MicroPolygon> mps = (objects.at(i))->getMicroPolygons();
        for (uint j = 0; j < mps.size(); j++) {
            MicroPolygon *screenSpace = projectMicroPolygon(mps.at(j));
            projectedMicroPolygons.push_back(screenSpace);
        }
    }
    // For each micropolygon, check what samples it intersects
    static int count = 0;
    for (uint i = 0; i < projectedMicroPolygons.size(); i++) {
        MicroPolygon *mp = projectedMicroPolygons.at(i);
        cout << "<a" << endl;
        float *f = mp->getBoundingBox();
        int px0_x = (int) (floor(f[0]));
        int px0_y = (int) (floor(f[1]));
        int px1_x = (int) (ceil(f[2]));
        int px1_y = (int) (ceil(f[3]));
        Vect point(0, 0, 0);
        for (int x = px0_x; x < px1_x; x++) {
            for (int y = px0_y; y < px1_y; y++) {
                // Loop over samples to get the right color
                unsigned char *mpColor = mp->getColor();
                // int sampledColor[3] = {0, 0, 0};
                // for (uint m = 0; m < this->m; m++) {
                //     for (uint n = 0; n < this->n * 2; n += 2) {
                //         point.setX(x + samples[2 * n + m]);
                //         point.setY(y + samples[2 * n + m + 1]);
                //         if (mp->intersects(point)) {
                //             sampledColor[0] += mpColor[0];
                //             sampledColor[1] += mpColor[1];
                //             sampledColor[2] += mpColor[2];
                //         }
                //     }
                // }
                // sampledColor[0] /= m*n;
                // sampledColor[1] /= m*n;
                // sampledColor[2] /= m*n;
                image.draw_point(x, y, mpColor);
            }
        }
        cout << projectedMicroPolygons.size() << ": " << count++ << endl;
    }

    for (uint i = 0; i < projectedMicroPolygons.size(); i++) {
        MicroPolygon *poly = projectedMicroPolygons.at(i);
        delete poly->a;
        delete poly->b;
        delete poly->c;
        delete poly->d;
        delete poly;
    }

    image.save_jpeg(name, 100);
}
