#include <stdlib.h>
#include <stdio.h>
#include "linkedlist.h"

void print_array_stats(double *array, int size) {
    double sum = 0, max = array[0];
    for (int i = 0; i < size; i++) {
        sum += array[i];
        if (max < array[i])
            max = array[i];
    }
    printf("Sum: %f\n", sum);
    printf("Avg: %f\n", sum / size);
    printf("Max: %f\n", max);
}

linked_list* new_linked_list(int size, int value) {
    linked_list *ll = (linked_list *) malloc(sizeof(linked_list));
    ll->next = NULL;
    ll->prev = NULL;
    ll->value = value;

    linked_list *next, *prev = ll;
    for (int i = 0; i < size - 1; i++) {
        next = (linked_list *) malloc(sizeof(linked_list));
        next->next = NULL;
        next->prev = prev;
        next->value = value;
        prev->next = next;
        prev = next;
    }
    return ll;
}

void print_linked_list(linked_list *ll, int horizontal, int direction) {
    char seperator = horizontal ? ' ' : '\n';

    if (direction) {
        linked_list *l = ll;
        do {
            printf("%d%c", l->value, seperator);
            l = l->next;
        } while (l);
        if (horizontal)
            printf("\n");
    } else {
        linked_list *l = ll;
        while (l->next) {
            l = l->next;
        }
        do {
            printf("%d%c", l->value, seperator);
            l = l->prev;
        } while (l);
        if (horizontal)
            printf("\n");
    }
}

int sum_linked_list(linked_list *ll) {
    int sum = 0;
    do {
        sum += ll->value;
        ll = ll->next;
    } while (ll);
    return sum;
}

void insert_linked_list(linked_list *ll, int pos, int value) {
    if (pos == 0) {
        linked_list *l = new_linked_list(1, ll->value);
        l->prev = ll;
        l->next = ll->next;
        ll->next->prev = l;
        ll->value = value;
        ll->next = l;
    } else {
        linked_list *temp = ll;
        while (pos > 0 && temp->next != NULL) {
            temp = temp->next;
            pos -= 1;
        }
        linked_list *l = new_linked_list(1, value);
        if (pos > 0) {
            temp->next = l;
            l->prev = temp;
        } else {
            l->prev = temp->prev;
            l->prev->next = l;
            l->next = temp;
            temp->prev = l;
        }
    }
}

void merge_linked_list(linked_list *a, linked_list *b) {
    int pos = 1;
    while (b) {
        insert_linked_list(a, pos, b->value);
        pos += 2;
        b = b->next;
    }
}

void destroy_linked_list(linked_list *ll) {
    linked_list *prev = ll;
    do {
        ll = ll->next;
        free(prev);
        prev = ll;
    } while (ll);
    free(prev);
}


int main(int argc, char **argv) {
    // Array statistics
    double array[5] = { 2.0, 3.89, -3.94, 10.1, 0.88 };
    print_array_stats(array, 5);

    // Creating liked list with 3 3s and 4 4s
    linked_list *ll3 = new_linked_list(3, 3);
    linked_list *ll4 = new_linked_list(4, 4);

    // Should print: "3 3 3"
    print_linked_list(ll3, 1, 1);

    // Inserting a 5 at the 1st position
    insert_linked_list(ll3, 1, 5);

    // Should print "3 5 3 3"
    print_linked_list(ll3, 1, 1);

    // Printing backwards, should print: "3 3 5 3"
    print_linked_list(ll3, 1, 0);

    // Merging the linked lists
    merge_linked_list(ll3, ll4);

    // Printing the result, should print: "3 4 5 4 3 4 3 4"
    print_linked_list(ll3, 1, 1);

    // Summing the elements, should be 30
    printf("Sum: %d\n", sum_linked_list(ll3));

    // Freeing the memory of ll3
    destroy_linked_list(ll3);
}
