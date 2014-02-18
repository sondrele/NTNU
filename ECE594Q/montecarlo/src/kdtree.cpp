#include "kdtree.h"

/*********************************
* KDTree
*********************************/
Node::Node() {
    leaf = false;
    left = NULL;
    right = NULL;
}

Node::Node(Node *l, Node *r) {
    left = l;
    right = r;
}

Node::~Node() {
    if (left != NULL) {
        delete left;
    }

    if (right != NULL) {
        delete right;
    }
}

bool Node::isLeaf() {
    return leaf;
}

void Node::setLeaf(bool l) {
    leaf = l;
}

int Node::getNumShapes() {
    return (int) shapes.size();
}

Shape * Node::getShape(uint i) {
    return shapes[i];
}

/*********************************
* KDTree
*********************************/
KDTree::KDTree() {
    dims = 3;
    shapesPerLeaf = 2;
    
    comparators = new comparator[dims];
    comparators[0] = Shape::CompareX;
    comparators[1] = Shape::CompareY;
    comparators[2] = Shape::CompareZ;

    root = NULL;
}

KDTree::~KDTree() {
    if (comparators != NULL) {
        delete [] comparators;
    }

    if (root != NULL) {
        delete root;
    }
}

void KDTree::setDims(int d) {
    dims = d;
}
void KDTree::setShapesPerLeaf(int s) {
    shapesPerLeaf = s;
}

Node * KDTree::buildTree(std::vector<Shape *> shapes) {
    int depth = 0;
    root = buildSubTree(shapes, depth);
    return root;
}

Node * KDTree::buildSubTree(std::vector<Shape *> shapes, int depth) {
    int size = (int) shapes.size();
    int axis = depth % dims; // 0-2

    // Construct leaf node containing shapes
    if (size <= shapesPerLeaf) {
        Node *leaf = new Node();
        leaf->shapes = shapes;
        leaf->leaf = true;
        leaf->axis = axis;
        return leaf;
    }
    // Otherwise split the shapes into two subsets, and divide them amongst
    // the left and right child nodes

    // Sort shapes based on the current axis
    std::sort(shapes.begin(), shapes.end(), comparators[axis]);

    // Find the median 
    int median = size / 2;
    std::vector<Shape *>::iterator mid = shapes.begin() + median;

    // Construct tree node
    Node *treeNode = new Node();
    Shape *s = *mid;
    treeNode->axis = axis;
    treeNode->point = s->getBBox().getLowerLeft();

    // Construct left child node
    std::vector<Shape *> lShapes(shapes.begin(), mid);
    treeNode->left = buildSubTree(lShapes, depth + 1);

    // Construct right child node
    std::vector<Shape *> rShapes(mid, shapes.end());
    treeNode->right = buildSubTree(rShapes, depth + 1);

    return treeNode;
}

Intersection KDTree::searchTree(Node *n, Ray r) {
    if (n->leaf) {
        Intersection ins;

        for (std::vector<Shape *>::iterator i = n->shapes.begin(); i != n->shapes.end(); ++i)
        {
            Intersection in = (*i)->intersects(r);
            if (in.hasIntersected()) {
                ins = in;
            }
        }
        return ins;
    } else {
        /* 
        Hvis axis er X, så har alle shapes blitt sortert mhp x,
        dvs at de i venstre node er de med minst x-verdier, hvis man da
        sjekker om ray går gjennom X-planet dannet fra punktet i noden
        */

        return Intersection();
    }
}

Node * KDTree::getRoot() {
    return root;
}
