#ifndef _CAMERA_H_
#define _CAMERA_H_

class Camera {
private:
    Vect pos;           // E
    Vect viewDir;       // V
    Vect orthoUp;       // Ù

    float focalDist;
    float verticalFOV;

public:
    Vect getPos() const { return pos;}
    void setPos(Vect d) { pos = d;}
    float getX() { return pos.getX(); }
    float getY() { return pos.getY(); }
    float getZ() { return pos.getZ(); }
    Vect getViewDir() { return viewDir;}
    void setViewDir(Vect d) { viewDir = d;}
    Vect getOrthoUp() { return orthoUp;}
    void setOrthoUp(Vect d) { orthoUp = d;}
    float getFocalDist() { return focalDist;}
    void setFocalDist(float f) { focalDist = f;}
    float getVerticalFOV() { return verticalFOV;}
    void setVerticalFOV(float f) { verticalFOV = f;}
};

#endif // _CAMERA_H_