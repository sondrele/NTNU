#include "progress.h"

Progress::Progress() {
    goal = 0;
    current = 0;
    percentage = 0;
}

void Progress::setGoal(int g) {
    goal = g;
}

void Progress::tick() {
    current += 1;
    update();
}

void Progress::update() {
    int p = (int) ((float) current / (float) goal * 100);
    if (p > percentage) {
        percentage = p;
        if (percentage % 5 == 0)
            std::cout << percentage << "%" << std::endl;
    }
}
