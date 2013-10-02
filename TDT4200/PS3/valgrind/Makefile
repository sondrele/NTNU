
test01: test01.c
	gcc -std=c99 -g test01.c -o test01
	valgrind --leak-check=full ./test01

test02: test02.c
	gcc -std=c99 -g $< -o test02
	valgrind --tool=exp-dhat ./test02

test03: test03.c
	gcc-4.6 -g $< -o t03
	gcc-4.6 -g -O3 $< -o to3
	gcc-4.6 -g -Wall $< -o twall3
	gcc-4.6 -g -Wall -O3 $< -o twallo3
	gcc-4.6 -g -Wall -O1 $< -o twallo1


clean:
	rm test03
	rm test02
	rm test01