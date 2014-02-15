#include "kdtree.h"

Node::Node() {
    left = NULL;
    right = NULL;
}

Node::Node(Node *l, Node *r) {
    left = l;
    right = r;
}

KDTree::KDTree(int k) {
    dims = k;
}

void KDTree::buildTree(std::vector<Shape *> shapes) {
    (void*) shapes[0];
}
