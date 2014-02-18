#ifndef _KD_TREE_H_
#define _KD_TREE_H_

#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#include "ray.h"
#include "intersection.h"
#include "rayscene_shapes.h"

class KDTree;

class Node {
friend class KDTree;
private:
    bool leaf;
    int axis;
    Vect point;
    Node *left;
    Node *right;
    std::vector<Shape *> shapes;

public:
    Node();
    Node(Node *, Node *);
    ~Node();

    bool isLeaf();
    void setLeaf(bool);
    Node * getLeft() { return left; }
    Node * getRight() { return right; }
    Vect getPoint() { return point; }
    int getNumShapes();
    Shape * getShape(uint);
};

typedef bool (*comparator)(Shape *, Shape *);

class KDTree {
private:
    int dims;
    int shapesPerLeaf;
    comparator *comparators;
    Node *root;

public:
    KDTree();
    ~KDTree();

    void setDims(int);
    void setShapesPerLeaf(int);
    Node * buildTree(std::vector<Shape *>);
    Node * buildSubTree(std::vector<Shape *>, int);
    Intersection searchTree(Node *n, Ray r);
    Node * getRoot();
};

#endif  // _KD_TREE_H_
