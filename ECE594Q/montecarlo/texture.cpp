#include "texture.h"

int main(int argc, char const *argv[])
{
    Texture t("test1.bmp");
    t.getTexel(0.5f, 0.5f);
    t.getTexel(0, 0);
    return 0;
}
