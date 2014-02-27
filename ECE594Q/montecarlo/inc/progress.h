#ifndef _PROGRESS_H_
#define _PROGRESS_H_

#include <iostream>

class Progress {
private:
    int goal;
    int current;
    int percentage;

public:
    Progress();
    void setGoal(int);
    void tick();
    void update();
};

#endif