#ifndef _MATERIAL_H_
#define _MATERIAL_H_

class Texture {
private:
    uint width;
    uint height;
    

public:
    Texture();
    Texture(uint, uint);

    void getTexel(uint, uint);
    void setTexture(std::string);

};

#endif // _MATERIAL_H_