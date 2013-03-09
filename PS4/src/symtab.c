#include "symtab.h"

#define STREAM_DATA 				".data\n"
#define STREAM_INTEGER 				".INTEGER: .string \"%%d \"\n"
#define STREAM_STRING 				".STRING%d: .string %s\n"
#define STREAM_GLOBL 				".globl main\n"

void print_stackframe ( void );

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
	if ( scopes_index == -1 ) {
		scopes = malloc( scopes_size * sizeof(void *) );
		scope_add ();
	} else if ( scopes_index == scopes_size ) {
		scopes_size *= 2;
		scopes = realloc ( scopes, sizeof(void *) * scopes_size );
	}

	if ( values_index == -1 )
		values = malloc( values_size * sizeof(void *) );
	else if ( values_index == values_size ) {
		values_size *= 2;
		values = realloc ( values, sizeof(void *) * values_size );	
	}	

	if ( strings_index == -1 )
		strings = malloc( strings_size * sizeof(char *) );
	else if ( strings_index == strings_size ) {
		strings_size *= 2;
		strings = realloc ( strings, sizeof(void *) * strings_size );
	}
}


void symtab_finalize ( void ) {
	for ( int i = scopes_index; i >= 0; i-- )
		free ( scopes[i] );
	free ( scopes );

	for ( int i = values_index; i >= 0; i-- )
		free ( values[i] );
	free ( values );

	for ( int i = strings_index; i >= 0; i-- )
		free ( strings[i] );
	free ( strings );
}


int32_t strings_add ( char *str ) {
	strings_index += 1;
	if ( strings_index == strings_size )
		symtab_init ();
	
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


void scope_add ( void ) {
	scopes_index += 1;
	if ( scopes_index == scopes_size ) 
		symtab_init ();

	scopes[scopes_index] = ght_create ( HASH_BUCKETS );
}


void scope_remove ( void ) {
	hash_t *top_scope = scopes[scopes_index];
	scopes_index -= 1;

	// ght_iterator_t iterator;
	// const void *p_key;
	// void *p_entry;
	// for ( p_entry = ght_first ( top_scope, &iterator, &p_key ); 
	// 		p_entry; p_entry = ght_next ( top_scope, &iterator, &p_key ) ) {
	// 	free(p_entry);
	// }

	ght_finalize ( top_scope );
}


void symbol_insert ( char *key, symbol_t *value ) {
	// Keep this for debugging/testing
	#ifdef DUMP_SYMTAB
		fprintf ( stderr, "Inserting (%s,%d)\n", key, value->stack_offset );
	#endif

	values_index += 1;
	if ( values_index == values_size )
		symtab_init ();

	if ( value->depth == -1 )
		value->depth = values_index;
	
	ght_insert ( scopes[scopes_index], value, strlen ( key ), key );
	values[values_index] = value;
}


symbol_t *symbol_get ( char *key ) {
	symbol_t *result = NULL;
	int i = scopes_index;

	while ( result == NULL ) {
		result = (symbol_t *) ght_get ( scopes[i], strlen ( key ), key );
		i -= 1;
	}

	// Keep this for debugging/testing
	#ifdef DUMP_SYMTAB
	    if ( result != NULL )
	        fprintf ( stderr, "Retrieving (%s,%d)\n", key, result->stack_offset );
	#endif

	return result;
}

void print_stackframe ( void ) {
	for ( int i = 0; i < scopes_index; i++ ) {
		printf("SCOPE %d\n", i);
		hash_t *scope = scopes[i];

		ght_iterator_t iterator;
		const void *p_key;
		void *p_entry;
		int j = 0;
		for(p_entry = ght_first(scope, &iterator, &p_key); p_entry; p_entry = ght_next(scope, &iterator, &p_key)) {
			printf("\tVALUE %d: %s\n", j++, (char *)p_key);
		}
		printf("\n");
	}
}