#include "tree.h"

void flatten_list ( node_t* node );
void evaluate_expression ( node_t* node );
void replace_node ( node_t* node, node_t* other );

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


node_t *node_init ( node_t *nd, nodetype_t type, void *data, uint32_t n_children, ... ) {
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
            (node->children)[i] = va_arg ( args, node_t *);
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


node_t *simplify_tree ( node_t *node ){

    if ( node != NULL ){

        // Recursively simplify the children of the current node
        for ( uint32_t i=0; i < node->n_children; i++ ){
            node->children[i] = simplify_tree ( node->children[i] );
        }

        // After the children have been simplified, we look at the current node
        // What we do depend upon the type of node
        switch ( node->type.index )
        {
            // These are lists which needs to be flattened. Their structure
            // is the same, so they can be treated the same way.
            case FUNCTION_LIST: case STATEMENT_LIST: case PRINT_LIST:
            case EXPRESSION_LIST: case VARIABLE_LIST:
                flatten_list ( node );
                break;

            // Declaration lists should also be flattened, but their stucture is sligthly
            // different, so they need their own case
            case DECLARATION_LIST:
                // First do a check to remove the nil-node, afterwards this can be handled 
                // the same way as the other lists
                if ( node->children[0] == NULL ) {
                    for (int i = 1; i < node->n_children; i++ )
                        node->children[i-1] = node->children[i];
                    node->n_children -= 1;
                    node_finalize (node->children[node->n_children]);
                }
                flatten_list ( node );
                break;
            
            // These have only one child, so they are not needed
            case STATEMENT: case PARAMETER_LIST: case ARGUMENT_LIST:
                if ( node != NULL ) {
                    node_t *child = node->children[0];
                    replace_node ( node, child );
                }
                break;

            // Expressions where both children are integers can be evaluated (and replaced with
            // integer nodes). Expressions whith just one child can be removed (like statements etc above)
            case EXPRESSION:
                evaluate_expression ( node ); 
                break;
        }
    }
    return node;
}

void flatten_list ( node_t *node ) {
    if ( node != NULL && node->children[0]->type.index == node->type.index) {
        node_t *child_list = node->children[0]; 
        int n_children = child_list->n_children + 1;

        node_t **children = malloc(sizeof(node_t *) * n_children);
        for ( int i = 0; i < child_list->n_children; i++ )
            children[i] = child_list->children[i];

        children[n_children-1] = node->children[1];
        node->n_children = n_children;
        node->children = children;
    }
}

void evaluate_expression ( node_t *node ) {
    if ( node->n_children == 1 ) {
        // Handle unary minus when the child is a number
        if ( node->data != NULL && *((char *) node->data) == '-' 
                && node->children[0]->type.index == INTEGER ) {
            int *data = (int *)node->children[0]->data;
            *data = -(*data);
            node->children[0]->type = integer_n;
            replace_node (node, node->children[0] );
        } 
        // Replace the node with it's child if it is not unary minus
        else if ( node->data == NULL || *((char *) node->data) != '-')
            replace_node (node, node->children[0] );    
    } 

    for ( int i = 0; i < node->n_children; i++ ) {
        if ( node->children[0]->type.index == INTEGER 
                && node->children[1]->type.index == INTEGER ) {                
            char operator = *((char *) node->data);
            int l_val ,
                r_val, 
                result;

            node_t *l_child = node->children[0];
            node_t *r_child = node->children[1];
            l_val = *((int *) l_child->data);
            r_val = *((int *) r_child->data);
            node_finalize ( r_child );
            node_finalize ( l_child );

            switch ( operator ) {
            case '+':
                result = l_val + r_val;
                break;
            case '-':
                result = l_val - r_val;
                break;
            case '*':
                result = l_val * r_val;
                break;
            case '/':
                result = l_val / r_val;
                break;
            }

            node->type = integer_n;
            *((int *) node->data) = result;
            node->n_children = 0;
            node->children = NULL;
        }
    }
}

void replace_node ( node_t *node, node_t *other ) {
    if ( node != NULL && other != NULL ) {
        node->type = other->type;
        node->data = other->data;
        node->entry = other->entry;
        node->n_children = other->n_children;
        node->children = other->children;   
        free ( other );
    }
}