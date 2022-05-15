#ifndef TREE_H_
#define TREE_H_

#include "data.h"

typedef struct TreeNode {
    struct TreeNode *left, *right;
    int feature_id;
    union Info {
        double leaf_value;
        double split_cond;
    } info;
} TreeNode;

typedef struct {
    TreeNode *root;
    int max_depth;
    double gamma, lambda;
} XGBoostTree;

void fitTree(Data *Xy, GradientPair *gpair, XGBoostTree *tree);

void predictTree(Data *Xy, double *outy, XGBoostTree *tree);

void printTree(XGBoostTree *tree);
#endif