#include "rpn.h"
#include<malloc.h>
#include<stdio.h>
#include<stdlib.h>

RpnCalc newRpnCalc(){
    // TODO initialize new RpnCalc
    RpnCalc calc = {0, 0, NULL };
    return calc;
}

void push(RpnCalc* rpnCalc, double n){
    // TODO push n on stack, expand stack if neccesary
    if(rpnCalc->size >= rpnCalc->top) {
        rpnCalc->stack = (double *) realloc(rpnCalc->stack, sizeof(double)*(rpnCalc->size + 10));
        rpnCalc->size += 10;
    }
    rpnCalc->top += 1;
    rpnCalc->stack[rpnCalc->top] = n;
}

void performOp(RpnCalc* rpnCalc, char op){
    // TODO perform operation
    if(rpnCalc->top > 1) {
        rpnCalc->top -= 1;
        int pos = rpnCalc->top;
        double s = rpnCalc->stack[pos+1];
        double t = rpnCalc->stack[pos];
        switch(op) {
        case '+':
            rpnCalc->stack[pos] = t + s;
            break;
        case '-':   
            rpnCalc->stack[pos] = t - s;
            break;
        case '*':
            rpnCalc->stack[pos] = t * s;
            break;
        case '/':
            rpnCalc->stack[pos] = t / s;
            break;            
        }
    }
}

double peek(RpnCalc* rpnCalc){
    // TODO return top element of stack
    if(rpnCalc->stack == NULL)
        return 0;
    return rpnCalc->stack[rpnCalc->top];
}

