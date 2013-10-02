#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

int main(int argc, char *argv[])
{
	char *mem = malloc(128);
	int i;
	for(i = 0; i < 32; i++)
		mem[i] = i;
	for(i = 0; i < 32; i++)
		mem[i+64] = mem[i];	


	mem[0] = 'H';
	mem[1] = 'e';
	mem[2] = 'i';

	printf("%d\n", mem[127]);
	printf("%d\n", mem[128]);
	printf("%d\n", mem[129]);

	return 0;
}
