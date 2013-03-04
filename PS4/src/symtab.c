#include "symtab.h"

#define STREAM_DATA 				".data\n"
#define STREAM_INTEGER 				".INTEGER: .string \"%%d \"\n"
#define STREAM_STRING 				".STRING%d: .string \"%s\"\n"
#define STREAM_GLOBL 				".globl main\n"

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
		// for ( int i = 0; i < scopes_size; i++ )
		// 	scopes[i] = malloc( hash_t );
	} else if ( scopes_index == scopes_size ) {
		scopes_size *= 2;
		scopes = realloc ( scopes, sizeof(void *) * scopes_size );
		// for ( int i = scopes_index; i < scopes_size; i++ )
		// 	scopes[i] = malloc( hash_t );
	}

	if ( values_size == -1 )
		values = malloc( values_size * sizeof(void *) );
	else if ( values_index == values_size ) {
		values_size *= 2;
		values = realloc ( values, sizeof(void *) * values_size );	
	}	

	if ( strings_size == -1 )
		strings = malloc( strings_size * sizeof(void *) );
	else if ( strings_index == strings_size ) {
		strings_size *= 2;
		strings = realloc ( strings, sizeof(void *) * strings_size );
	}
}


void symtab_finalize ( void ) {

}


int32_t strings_add ( char *str ) {
	strings_index += 1;
	if ( strings_index == strings_size )
		symtab_init ( void );
	
	strings[strings_index] = str;
	return strings_index;
}


void strings_output ( FILE *stream ) {
	if ( stream != NULL ) {
		fprintf ( *stream, STREAM_DATA );
		fprintf ( *stream, STREAM_INTEGER );
		for ( int i = 0; i < strings_index; i++ )
			fprintf ( stream, STREAM_STRING, i, strings[i] );
		
		fprintf ( *stream, STREAM_GLOBL );
	}
}


void scope_add ( void ) {
	scopes_index += 1;
	if ( scopes_index == scopes_size ) 
		symtab_init ( void );

	scopes[i] = malloc( sizeof( hash_t ) );
}


void scope_remove ( void ) {
	scopes_index -= 1;
}


void symbol_insert ( char *key, symbol_t *value ) {
	// Keep this for debugging/testing
	#ifdef DUMP_SYMTAB
		fprintf ( stderr, "Inserting (%s,%d)\n", key, value->stack_offset );
	#endif
}


symbol_t *symbol_get ( char *key ) {
	symbol_t *result = NULL;
	// Keep this for debugging/testing
	#ifdef DUMP_SYMTAB
	    if ( result != NULL )
	        fprintf ( stderr, "Retrieving (%s,%d)\n", key, result->stack_offset );
	#endif
}
