#include "tree.h"
#include "symtab.h"

#define VSL_PTR_SIZE		4;

void add_functions_to_symtab ( node_t *function_list_n );
void add_parameters_to_scope ( node_t *function_n );
int add_variables_to_scope ( node_t *variable_list_n, int stack_offset );
void add_var_to_scope ( node_t *variable_n, int stack_offset );
void add_text ( node_t *text_n );
void traverse_children ( node_t *root );


#ifdef DUMP_TREES
void
node_print ( FILE *output, node_t *root, uint32_t nesting ) {
	if ( root != NULL )
	{
		fprintf ( output, "%*c%s", nesting, ' ', root->type.text );
		if ( root->type.index == INTEGER )
			fprintf ( output, "(%d)", *((int32_t *)root->data) );
		if ( root->type.index == VARIABLE || root->type.index == EXPRESSION )
		{
			if ( root->data != NULL )
				fprintf ( output, "(\"%s\")", (char *)root->data );
			else
				fprintf ( output, "%p", root->data );
		}
		fputc ( '\n', output );
		for ( int32_t i=0; i<root->n_children; i++ )
			node_print ( output, root->children[i], nesting+1 );
	}
	else
		fprintf ( output, "%*c%p\n", nesting, ' ', root );
}
#endif


node_t *node_init ( node_t *nd, nodetype_t type, void *data, uint32_t n_children, ... ) {
	va_list child_list;
	*nd = (node_t) { type, data, NULL, n_children,
		(node_t **) malloc ( n_children * sizeof(node_t *) )
	};
	va_start ( child_list, n_children );
	for ( uint32_t i = 0; i < n_children; i++ )
		nd->children[i] = va_arg ( child_list, node_t * );
	va_end ( child_list );
	return nd;
}


void node_finalize ( node_t *discard ) {
	if ( discard != NULL ) {
		if ( discard->data != NULL )
			free ( discard->data );
		free ( discard->children );
		free ( discard );
	}
}


void destroy_subtree ( node_t *discard ) {
	if ( discard != NULL ) {
		for ( uint32_t i = 0; i < discard->n_children; i++ )
			destroy_subtree ( discard->children[i] );
		node_finalize ( discard );
	}
}


void bind_names ( node_t *root ) {
	if (root == NULL) 
		return;

	switch (root->type.index) {
		case FUNCTION_LIST: {
			add_functions_to_symtab ( root );
			traverse_children ( root );
			break;
		}
		case FUNCTION: {
			scope_add ();
			add_parameters_to_scope ( root );
			if ( root->children[2]->type.index == BLOCK) {
				bind_names ( root->children[2]->children[0] );
				bind_names ( root->children[2]->children[1] );
			} else
				fprintf(stderr, 
					"Expected BLOCK-statement after FUNCTION \"%s\"\n", (char *) root->data);
			scope_remove ();
			break;
		}
		case BLOCK: {
			scope_add ();
			traverse_children ( root );
			scope_remove ();
			break;
		}
		case DECLARATION_LIST: {
			int stack_offset = -VSL_PTR_SIZE;
			for ( int i = 0; i < root->n_children; i++ ) {
				node_t *declaration_n = root->children[i];
				stack_offset = add_variables_to_scope ( declaration_n, stack_offset );
			}
			break;
		}		
		case VARIABLE: {
			symbol_t *var = symbol_get ( (char *) root->data );
			if ( var != NULL ) 
				root->entry = var;
			else
				fprintf(stderr, "The variable \"%s\" is not declared.\n", (char *) root->data);
			traverse_children ( root );
			break;
		}
		case TEXT: {
			add_text ( root );
			traverse_children ( root );
			break; 
		}
		default:
		traverse_children ( root );
	}
}


void add_functions_to_symtab ( node_t *function_list_n ) {
	for ( int i = 0; i < function_list_n->n_children; i++ ) {
		node_t *func = function_list_n->children[i];
		symbol_t *var = malloc( sizeof( symbol_t ) );
		*var = (symbol_t) { 0, 0, (char *) func->children[0]->data };
		
		symbol_insert ( var->label, var );
	}
}


void add_parameters_to_scope ( node_t *function_n ) {
	node_t *params = function_n->children[1];

	if ( params != NULL ) {
		int stack_offset = ( 1 + params->n_children ) * VSL_PTR_SIZE;
	
		for ( int i = 0; i < params->n_children; i++ ) {
			node_t *variable_n = params->children[i];
			add_var_to_scope ( variable_n, stack_offset );
			stack_offset -= VSL_PTR_SIZE;			
		}
	}
}


int add_variables_to_scope ( node_t *declaration_list_n, int stack_offset ) {
	node_t *variable_list_n = declaration_list_n->children[0];

	for ( int i = 0; i < variable_list_n->n_children; i++ ) {
		node_t *variable_n = variable_list_n->children[i];
		add_var_to_scope ( variable_n, stack_offset );
		stack_offset -= VSL_PTR_SIZE;
	}
	return stack_offset;
}


void add_var_to_scope ( node_t *variable_n, int stack_offset ) {
	symbol_t *var = malloc( sizeof(symbol_t) );
	*var = (symbol_t) { stack_offset, -1, NULL };
	symbol_insert ( (char *) variable_n->data, var );
}


void add_text ( node_t *text_n ) {
	int *str_ptr = malloc ( sizeof(int *) );
	*str_ptr = strings_add ( (char *) text_n->data );
	text_n->data = str_ptr;
}


void traverse_children ( node_t *root ) {
	for ( int i = 0; i < root->n_children; i++ )
		bind_names ( root->children[i] );
}
