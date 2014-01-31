#ifndef _UTILS_H_
#define _UTILS_H_

#include <exception>
#include "Matrix.h"

#define U_PI 3.141592654

class Utils {
public:
    static float Sign(Vect, Vect, Vect);
    static bool PointInTriangle(Vect, Vect, Vect, Vect);
};

#endif // _UTILS_H_
