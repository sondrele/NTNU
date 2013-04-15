#include <stdio.h>

int main () {
	precedence();
	return 0;
}

int precedence () {
	int a, b, c, d;
	a = 2;
	b = 3;
	c = 1;
	d = a * (b - c);
	printf("2*(3-1) = %d\n", d);
	d = a*b-c;
	printf("2*3-1 = %d\n", d);
	return 0;
}