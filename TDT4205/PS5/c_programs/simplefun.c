#include <stdio.h>

int main () {
	funcall();
	return 0;
}

void funcall (){
	int x = my_function ( 5, 10 );
}

int my_function ( int s, int t ) {
	printf("Parameter s is %d, t is %d\n", s, t);
	return 0;
}