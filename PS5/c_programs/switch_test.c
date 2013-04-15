#include <stdio.h>
#include <stdlib.h>

int main() {
	void *a = malloc(1);
	a[0] = 'H';
	printf("%s\n", (char *)a);
	switch ( *(char *)a ) {
		case 'H':
		printf("H\n");
		break;
		case 'e':
		printf("e\n");
		break;
		case 'i':
		printf("i\n");
		break;
		default:
		printf("Error\n");

	}
	return 0;
}