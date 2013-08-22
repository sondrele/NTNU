/* runner.c */

#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include "linkedlist.h"
#include "test_runner.h"

void error_msg(void *actual, void *expected, int type) {
    switch (type) {
    case CHAR:
        printf("%3$sExpected %4$s%2$s%3$s, but was %4$s%1$s%3$s\n", (char*)actual, (char*)expected, red, cyan);
        return;
    case INT:
        printf("%3$sExpected %4$s%2$d%3$s, but was %4$s%1$d%3$s\n", (int)actual, (int)expected, red, cyan);
        return;
    }
}

void stub_printf() {
    stdout_fd = dup(STDOUT_FILENO);
    freopen(".temp_printf", "w", stdout);
}

char *read_printf() {
    char *file_contents;
    int input_file_size;
    FILE *input_file = fopen(".temp_printf", "rb");
    fseek(input_file, 0, SEEK_END);
    input_file_size = ftell(input_file);
    rewind(input_file);
    file_contents = malloc(input_file_size);
    fread(file_contents, 1, input_file_size, input_file);
    fclose(input_file);
    if (file_contents[input_file_size] != '\0') {
        file_contents[input_file_size] = '\0';
    }
    return file_contents;
}

void restore_printf() {
    fclose(stdout);
    dup2(stdout_fd, STDOUT_FILENO);
    stdout = fdopen(STDOUT_FILENO, "w");
    close(stdout_fd);
}

void print_summary(int result) {
    if (result == 0)
        printf("%sPASSED\n", green);
    printf("Tests run: %d\n", tests_run);
}


/* UNIT TESTS */

int linked_list_has_right_value() {
    linked_list *ll = new_linked_list(1, 0);
    ASSERT_EQUAL_INT(ll->value, 0);
    ASSERT_EQUAL_INT(ll->prev, 0);
    ASSERT_EQUAL_INT(ll->next, 0);
    return 0;
}

int linked_list_has_next() {
    linked_list *ll = new_linked_list(2, 0);
    ASSERT_TRUE(ll->next);
    ASSERT_EQUAL_INT(ll->next->value, 0);
    ASSERT_EQUAL_INT(ll->next->prev, ll);
    return 0;
}

int can_insert_to_start_of_linked_list() {
    linked_list *ll = new_linked_list(3, 0);
    insert_linked_list(ll, 0, 3);

    ASSERT_EQUAL_INT(ll->prev, NULL);
    ASSERT_EQUAL_INT(ll->next->value, 0);
    ASSERT_EQUAL_INT(ll->next->prev, ll);
    ASSERT_EQUAL_INT(ll->value, 3);
    return 0;
}

int can_insert_in_middle_of_linked_list() {
    linked_list *ll = new_linked_list(3, 0);
    insert_linked_list(ll, 2, 3);

    linked_list *inserted = ll->next->next;
    ASSERT_EQUAL_INT(ll->prev, NULL);
    ASSERT_EQUAL_INT(ll->value, 0);
    ASSERT_EQUAL_INT(ll->next->value, 0);
    ASSERT_EQUAL_INT(inserted->prev, ll->next);
    ASSERT_EQUAL_INT(inserted->next->prev, inserted);
    ASSERT_EQUAL_INT(inserted->value, 3);
    return 0;
}

int can_insert_at_end_of_linked_list() {
    linked_list *ll = new_linked_list(3, 0);
    insert_linked_list(ll, 3, 3);

    linked_list *last = ll->next->next->next;
    ASSERT_NOT_EQUAL_INT(last, NULL);
    ASSERT_EQUAL_INT(last->next, NULL);
    ASSERT_EQUAL_INT(last->value, 3);
    return 0;
}

int linked_list_can_be_printed_horizontally() {
    linked_list *ll = new_linked_list(3, 3);
    insert_linked_list(ll, 0, 0);
    stub_printf();
    print_linked_list(ll, 1, 1);
    restore_printf();

    char *actual = read_printf();
    char *expected = "0 3 3 3 \n";
    ASSERT_EQUAL_STRING(actual, expected);
    return 0;
}

int linked_list_can_be_printed_vertically() {
    linked_list *ll = new_linked_list(3, 3);
    stub_printf();
    print_linked_list(ll, 0, 1);
    restore_printf();

    char *actual = read_printf();
    char *expected = "3\n3\n3\n";
    ASSERT_EQUAL_STRING(actual, expected);
    return 0;
}

int linked_list_can_be_printed_backwards() {
    linked_list *ll = new_linked_list(3, 3);
    insert_linked_list(ll, 4, 1);
    stub_printf();
    print_linked_list(ll, 1, 0);
    restore_printf();

    char *actual = read_printf();
    char *expected = "1 3 3 3 \n";
    ASSERT_EQUAL_STRING(actual, expected);
    return 0;
}

int sum_of_linked_list() {
    linked_list *ll = new_linked_list(3, 3);
    int sum = sum_linked_list(ll);
    ASSERT_EQUAL_INT(sum, 9);
    return 0;
}

int sum_of_linked_list_after_insertion() {
    linked_list *ll = new_linked_list(3, 3);
    insert_linked_list(ll, 2, 9);
    int sum = sum_linked_list(ll);
    ASSERT_EQUAL_INT(sum, 18);
    return 0;
}

int linked_lists_can_be_merged() {
    linked_list *l1 = new_linked_list(3, 3);
    linked_list *l2 = new_linked_list(3, 0);
    merge_linked_list(l1, l2);

    ASSERT_EQUAL_INT(l1->value, 3);
    ASSERT_EQUAL_INT(l1->next->value, 0);

    linked_list *last = l1->next->next->next->next->next;
    ASSERT_NOT_EQUAL_INT(last, NULL);
    ASSERT_EQUAL_INT(last->value, 0);
    ASSERT_EQUAL_INT(last->prev->value, 3);
    return 0;
}

int linked_list_can_be_destroyed() {
    linked_list *ll = new_linked_list(3, 3);
    destroy_linked_list(ll);
    
    return 0;
}

int all_tests() {
    _verify(linked_list_has_right_value);
    _verify(linked_list_has_next);
    
    _verify(can_insert_to_start_of_linked_list);
    _verify(can_insert_in_middle_of_linked_list);
    _verify(can_insert_at_end_of_linked_list);

    _verify(linked_list_can_be_printed_horizontally);
    _verify(linked_list_can_be_printed_vertically);
    _verify(linked_list_can_be_printed_backwards);

    _verify(sum_of_linked_list);
    _verify(sum_of_linked_list_after_insertion);

    _verify(linked_lists_can_be_merged);

    _verify(linked_list_can_be_destroyed);
    return 0;
}

int main(int argc, char **argv) {
    int result = all_tests();
    print_summary(result);
    return result != 0;
}
