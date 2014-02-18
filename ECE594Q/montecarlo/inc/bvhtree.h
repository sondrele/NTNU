#ifndef _BVH_TREE_H_
#define _BVH_TREE_H_

#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#include "ray.h"
#include "intersection.h"
#include "rayscene_shapes.h"

class BVHTree;

class BVHNode {
friend class BVHTree;
private:
    BVHNode *left;
    BVHNode *right;
    Shape *shape;
    BBox bbox;
    bool leaf;

public:
    BVHNode();
    BVHNode(BVHNode *, BVHNode *);
    ~BVHNode();

    void setBBox(BBox);
    BBox getBBox();
    void addShape(Shape *);
    Shape * getShape() { return shape; }

    bool isLeaf();
    void setLeaf(bool);
    BVHNode * getLeft() { return left; }
    BVHNode * getRight() { return right; }
};

typedef bool (*comparator)(Shape *, Shape *);

class BVHTree {
private:
    int dims;
    int shapesPerLeaf;
    comparator *comparators;
    BVHNode *root;

public:
    BVHTree();
    ~BVHTree();

    void setDims(int);
    void setShapesPerLeaf(int);
    BVHNode * buildTree(std::vector<Shape *>);
    BVHNode * buildSubTree(std::vector<Shape *>, int);
    Intersection intersects(Ray);
    Intersection searchTree(BVHNode *n, Ray r);
    BVHNode * getRoot();
};

#endif // _BVH_TREE_H_