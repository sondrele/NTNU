
typedef struct linked_list {
    struct linked_list *next;
    struct linked_list *prev;
    int value;
} linked_list;

void print_array_stats(double *array, int size);

linked_list* new_linked_list(int size, int value);

void print_linked_list(linked_list *ll, int horizontal, int direction);

int sum_linked_list(linked_list *ll);

void insert_linked_list(linked_list *ll, int pos, int value);

void merge_linked_list(linked_list *a, linked_list *b);

void destroy_linked_list(linked_list *ll);

void print_summary(int);
