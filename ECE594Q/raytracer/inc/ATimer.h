#ifndef TIMER_H
#define TIMER_H

#include <cstdio>
#include <ctime>
#include <unistd.h>

class Timer {
private:
    std::clock_t start, end, res;
public:
    Timer() {

    }
    
    void resetTimer(void) {
        start = 0;
        end = 0;
        res = 0;
    }

    void unpauseTimer(void) {
    }

    void pauseTimer(void) {
    }

    void startTimer(void) {
        start = std::clock();
    }

    void stopTimer(void) {
        end = std::clock();
        res = end - start;
    }

    void printTime(void) {
        printf("%lf\n", (double)res);
    }

    double getTime(void) const{
        return ((double) res);
    }

};



#endif      /* TIMER_H */