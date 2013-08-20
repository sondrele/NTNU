#include <stdlib.h>
#include <stdio.h>


typedef struct linked_list{
    struct linked_list* next;
    struct linked_list* prev;
    int value;
} linked_list;


void print_array_stats(double* array, int size){
}


linked_list* new_linked_list(int size, int value){
}


void print_linked_list(linked_list* ll, int horizontal, int direction){
}


int sum_linked_list(linked_list* ll){
}


void insert_linked_list(linked_list* ll, int pos, int value){
}


void merge_linked_list(linked_list* a, linked_list* b){
}

void destroy_linked_list(linked_list* ll){
}

    


int main(int argc, char** argv){

    //Array statistics
    double array[5] = {2.0, 3.89, -3.94, 10.1, 0.88};
    print_array_stats(array, 5);

    //Creating liked list with 3 3s and 4 4s
    linked_list* ll3 = new_linked_list(3,3);
    linked_list* ll4 = new_linked_list(4,4);

    //Should print: "3 3 3"
    print_linked_list(ll3, 1, 1);

    //Inserting a 5 at the 1st position
    insert_linked_list(ll3, 1, 5);

    //Should print "3 5 3 3"
    print_linked_list(ll3, 1, 1);

    //Printing backwards, should print: "3 3 5 3"
    print_linked_list(ll3, 1, 0);

    //Merging the linked lists
    merge_linked_list(ll3, ll4);

    //Printing the result, should print: "3 4 5 4 3 4 3 4"
    print_linked_list(ll3, 1,1);

    //Summing the elements, should be 30
    printf("Sum: %d\n", sum_linked_list(ll3));

    //Freeing the memory of ll3
    destroy_linked_list(ll3);
}
