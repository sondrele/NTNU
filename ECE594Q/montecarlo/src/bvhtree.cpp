#include "bvhtree.h"

/*********************************
* BVHNode
*********************************/
BVHNode::BVHNode() {
    leaf = false;
    left = NULL;
    right = NULL;
}

BVHNode::BVHNode(BVHNode *l, BVHNode *r) {
    leaf = false;
    left = l;
    right = r;
}

BVHNode::~BVHNode() {
    if (left != NULL) {
        delete left;
    }

    if (right != NULL) {
        delete right;
    }
}

void BVHNode::setBBox(BBox b) {
    bbox = b;
}

BBox BVHNode::getBBox() {
    return bbox;
}

void BVHNode::addShape(Shape *s) {
    shape = s;
    bbox = shape->getBBox();
}

bool BVHNode::isLeaf() {
    return left == NULL && right == NULL;
}

void BVHNode::setLeaf(bool l) {
    leaf = l;
}

/*********************************
* BVHTree
*********************************/
BVHTree::BVHTree() {
    dims = 3;
    shapesPerLeaf = 1;
    
    comparators = new comparator[dims];
    comparators[0] = Shape::CompareX;
    comparators[1] = Shape::CompareY;
    comparators[2] = Shape::CompareZ;

    root = NULL;
}

BVHTree::~BVHTree() {
    if (comparators != NULL) {
        delete [] comparators;
    }

    if (root != NULL) {
        delete root;
    }
}

BVHNode * BVHTree::buildTree(std::vector<Shape *> shapes) {
    if (root != NULL) {
        delete root;
    }

    int depth = 0;
    root = buildSubTree(shapes, depth);
    return root;
}

BVHNode * BVHTree::buildSubTree(std::vector<Shape *> shapes, int depth) {
    int size = (int) shapes.size();
    int axis = depth % dims; // 0-2

    // Construct leaf BVHNode containing shapes
    if (size == shapesPerLeaf) {
        BVHNode *leaf = new BVHNode();
        leaf->addShape(shapes[0]); // Add shape and init bbox
        leaf->leaf = true;
        return leaf;
    } else if (size == 0) {
        return NULL;
    }
    // Otherwise split the shapes into two subsets, and divide them amongst
    // the left and right child nodes

    // Sort shapes based on the current axis
    std::sort(shapes.begin(), shapes.end(), comparators[axis]);

    // Find the median 
    int median = size / 2;
    std::vector<Shape *>::iterator mid = shapes.begin() + median;

    // Construct tree BVHNode
    BVHNode *treeNode = new BVHNode();

    // Construct left child BVHNode
    std::vector<Shape *> lShapes(shapes.begin(), mid);
    treeNode->left = buildSubTree(lShapes, depth + 1);

    // Construct right child BVHNode
    std::vector<Shape *> rShapes(mid, shapes.end());
    treeNode->right = buildSubTree(rShapes, depth + 1);

    // Store the bbox for the treeNode
    treeNode->bbox = treeNode->left->bbox + treeNode->right->bbox;
    return treeNode;
}

Intersection BVHTree::intersects(Ray r) {
    return searchTree(root, r);
}

Intersection BVHTree::searchTree(BVHNode *n, Ray r) {
    if (n->leaf) {
        return n->shape->intersects(r);
    } else {
        if (n->bbox.intersects(r)) {
            Intersection i = searchTree(n->left, r);
            Intersection j = searchTree(n->right, r);

            if (i.hasIntersected() && j.hasIntersected()) {
                return i.getIntersectionPoint() < j.getIntersectionPoint() ? i : j;
            } else {
                return i.hasIntersected() ? i : j;
            }
        }
        return Intersection();
    }
}

BVHNode * BVHTree::getRoot() {
    return root;
}
