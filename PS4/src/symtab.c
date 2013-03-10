#include "symtab.h"

#define STREAM_DATA 				".data\n"
#define STREAM_INTEGER 				".INTEGER: .string \"%%d \"\n"
#define STREAM_STRING 				".STRING%d: .string %s\n"
#define STREAM_GLOBL 				".globl main\n"

void print_status ( void );
void print_values ( void );
void print_strings ( void );
void print_scopes ( void );

// static does not mean the same as in Java.
// For global variables, it means they are only visible in this file.

// Pointer to stack of hash tables 
static hash_t **scopes;

// Pointer to array of values, to make it easier to free them
static symbol_t **values;

// Pointer to array of strings, should be able to dynamically expand as new strings
// are added.
static char **strings;

// Helper variables for manageing the stacks/arrays
static int32_t scopes_size = 16, scopes_index = -1;
static int32_t values_size = 16, values_index = -1;
static int32_t strings_size = 16, strings_index = -1;


void symtab_init ( void ) {
	if ( scopes_index == -1 )
		scopes = malloc( scopes_size * sizeof(hash_t *) );

	if ( values_index == -1 )
		values = malloc( values_size * sizeof(symbol_t *) );
	
	if ( strings_index == -1 )
		strings = malloc( strings_size * sizeof(char *) );

	scope_add ();
}


void symtab_finalize ( void ) {
	// print_status ();
	scope_remove ();
	for ( int i = values_index; i >= 0; i-- ) {
		free ( values[i] );
	}
	free ( values );

	for ( int i = strings_index; i >= 0; i-- )
		free ( strings[i] );
	free ( strings );

	for ( int i = scopes_index; i >= 0; i-- )
		free ( scopes[i] );
	free ( scopes );

}


void scope_add ( void ) {
	scopes_index += 1;
	if ( scopes_index == scopes_size )  {
		void *tmp = realloc ( scopes, sizeof(hash_t *) * (scopes_size *= 2) );
		if ( tmp != NULL )
			scopes = tmp;
	}
	hash_t *scope = ght_create ( HASH_BUCKETS );
	scopes[scopes_index] = scope;
}


void scope_remove ( void ) {
	hash_t *top_scope = scopes[scopes_index--];
	
	// ght_iterator_t iterator;
	// const void *p_key;
	// void *p_entry;
	// for ( p_entry = ght_first ( top_scope, &iterator, &p_key ); 
	// 		p_entry; p_entry = ght_next ( top_scope, &iterator, &p_key ) ) {
	// //	printf("Freeing entry at key=%s\n", p_key);
	// 	free(p_entry);
	// }
	// print_scopes ();

	ght_finalize ( top_scope );
}


int32_t strings_add ( char *str ) {
	strings_index += 1;
	if ( strings_index == strings_size ) {
		void *tmp = realloc ( strings, sizeof(char *) * (strings_size *= 2) );
		if ( tmp != NULL )
			strings = tmp;
	}
	
	strings[strings_index] = str;
	return strings_index;
}


void strings_output ( FILE *stream ) {
	if ( stream != NULL ) {
		fprintf ( stream, STREAM_DATA );
		fprintf ( stream, STREAM_INTEGER );
		for ( int i = 0; i < strings_index + 1; i++ )
			fprintf ( stream, STREAM_STRING, i, strings[i] );
		
		fprintf ( stream, STREAM_GLOBL );
	}
}


void symbol_insert ( char *key, symbol_t *value ) {
	// Keep this for debugging/testing
	#ifdef DUMP_SYMTAB
		fprintf ( stderr, "Inserting (%s,%d)\n", key, value->stack_offset );
	#endif

	values_index += 1;
	if ( values_index == values_size ) {
		void *tmp = realloc ( values, sizeof(symbol_t *) * (values_size *= 2) );	
		if ( tmp != NULL )
			values = tmp;
	}


	if ( value->depth == -1 )
		value->depth = values_index;
	
	ght_insert ( scopes[scopes_index], value, strlen ( key ), key );
	values[values_index] = value;
}


symbol_t *symbol_get ( char *key ) {
	symbol_t *result = NULL;
	int i = scopes_index;

	while ( result == NULL )
		result = (symbol_t *) ght_get ( scopes[i--], strlen ( key ), key );

	// Keep this for debugging/testing
	#ifdef DUMP_SYMTAB
	    if ( result != NULL )
	        fprintf ( stderr, "Retrieving (%s,%d)\n", key, result->stack_offset );
	#endif

	return result;
}

void print_status ( void ) {
	print_scopes ();
	print_values ();
	print_strings ();
}

void print_scopes ( void ) {
	printf("-----SCOPES-----\n");
	for ( int i = 0; i < scopes_index+1; i++ ) {
		hash_t *t = scopes[i];
		printf("SCOPE%d: items=%d\n", i, t->i_items);
	}
	printf("---END SCOPES---\n");
}

void print_values ( void ) {
	printf("-----VALUES-----\n");
	for ( int i = 0; i < values_index+1; i++ ) {
		symbol_t *t = values[i];
		printf("VALUE%d: offset=%d, depth=%d, label=%s\n", i, t->stack_offset, t->depth, t->label);
	}
	printf("---END VALUES---\n");
}

void print_strings ( void ) {
	printf("-----STRINGS-----\n");
	for ( int i = 0; i < strings_index+1; i++ ) {
		char *s = strings[i];
		printf("STRING%d: %s\n", i, s);
	}
	printf("---END STRINGS---\n");
}