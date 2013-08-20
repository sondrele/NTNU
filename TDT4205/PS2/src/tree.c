#include "tree.h"
#include <stdarg.h>

#ifdef DUMP_TREES
void
node_print ( FILE *output, node_t *root, uint32_t nesting )
{
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


node_t * node_init ( node_t *nd, nodetype_t type, void *data, uint32_t n_children, ... ) {
    node_t *node = nd;
    node->type = type;
    node->data = data;
    node->n_children = n_children;
    node->entry = NULL;

    if ( n_children > 0 ) {
        node->children = malloc(sizeof(node_t*)*n_children);
        va_list args;
        va_start ( args, n_children );
        for ( int i = 0; i < node->n_children; i++ ) {
            (node->children)[i] = va_arg ( args, node_t* );
        }
        va_end ( args );  
    } else 
        node->children = NULL;

    return node;
}


void node_finalize ( node_t *discard ) {
    free ( discard->data );
    free ( discard->entry );
    free ( discard );
}


void destroy_subtree ( node_t *discard ){
    if ( discard == NULL )
        return;
    for ( int i = 0; i < discard->n_children; i++ )
        destroy_subtree ( (discard->children)[i] );
    node_finalize ( discard );
}
