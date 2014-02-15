#ifndef _KD_TREE_H_
#define _KD_TREE_H_

#include <cstdlib>
#include <string>
#include <vector>

#include "rayscene_shapes.h"

class Node {
private:
    Node *left;
    Node *right;
    std::vector<Shape *> shapes;

public:
    Node();
    Node(Node *, Node *);
};

class KDTree {
private:
    int dims;


public:
    KDTree(int);

    void buildTree(std::vector<Shape *>);
};

#endif  // _KD_TREE_H_
