#include "tree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"
#include "utils.h"
#define score(G, H) (((G) * (G)) / ((H) + tree->lambda))
#define weight(G, H) (-(G) / ((H) + tree->lambda))

TreeNode *splitNode(Data *Xy, Subset *index_subset, int cur_depth,
                    GradientPair *gpair, XGBoostTree *tree) {
    double best_score = 0, before_score, after_score;
    double G = 0, H = 1e-16;
    double best_split;
    int best_feature;
    TreeNode *ret = mallocOrDie(sizeof(TreeNode));
    for (int i = 0; i < Xy->n_example; i++)
        if (inSubset(i, index_subset)) {
            G += gpair[i].g;
            H += gpair[i].h;
        }
    if (cur_depth == tree->max_depth) {
        ret->left = NULL;
        ret->right = NULL;
        ret->info.leaf_value = weight(G, H);
        return ret;
    }
    before_score = score(G, H);
    Subset *next_subset = initSubset(index_subset->size, 0);
    Subset *best_next_subset = initSubset(index_subset->size, 0);
    for (int i = 0; i < Xy->n_feature; i++) {
        double G_L = 0, H_L = 0, G_R, H_R;
        resetSubset(next_subset);
        for (int j = 0; j < Xy->n_example; j++) {
            if (next_subset->cnt == index_subset->cnt - 1)
                break;
            int example_index = Xy->feature_blocks[i][j];
            if (!inSubset(example_index, index_subset))
                continue;
            addToSubset(example_index, next_subset);
            G_L += gpair[example_index].g;
            H_L += gpair[example_index].h;
            G_R = G - G_L;
            H_R = H - H_L;
            after_score = score(G_L, H_L) + score(G_R, H_R);
            if (after_score > best_score) {
                memcpy(best_next_subset->bitset, next_subset->bitset,
                       sizeof(char) * BITNSLOTS(next_subset->size));
                best_next_subset->cnt = next_subset->cnt;
                best_feature = i;
                best_split = Xy->X[example_index][i];
                best_score = after_score;
            }
        }
    }
    if (best_score > before_score + tree->gamma) {
        ret->feature_id = best_feature;
        ret->info.split_cond = best_split;
        ret->left = splitNode(Xy, best_next_subset, cur_depth + 1, gpair, tree);
        best_next_subset->cnt = index_subset->cnt - best_next_subset->cnt;
        for (int i = 0; i < BITNSLOTS(best_next_subset->size); i++)
            best_next_subset->bitset[i] =
                index_subset->bitset[i] & ~(best_next_subset->bitset[i]);
        ret->right =
            splitNode(Xy, best_next_subset, cur_depth + 1, gpair, tree);
    } else {
        ret->left = NULL;
        ret->right = NULL;
        ret->info.leaf_value = weight(G, H);
    }
    freeSubset(next_subset);
    freeSubset(best_next_subset);
    return ret;
}

void fitTree(Data *Xy, GradientPair *gpair, XGBoostTree *tree) {
    Subset *ss = initSubset(Xy->n_example, 1);
    tree->root = splitNode(Xy, ss, 0, gpair, tree);
    freeSubset(ss);
}

void predictTree(Data *Xy, double *outy, XGBoostTree *tree) {
    TreeNode *node = NULL;
    for (int i = 0; i < Xy->n_example; i++) {
        node = tree->root;
        while (!isLeaf(node)) {
            double t = Xy->X[i][node->feature_id];
            if (t <= node->info.split_cond)
                node = node->left;
            else
                node = node->right;
        }
        outy[i] = node->info.leaf_value;
    }
}
void printNode(TreeNode *node, int depth) {
    for (int i = 0; i < depth; i++) printf("\t");
    if (isLeaf(node))
        printf("leaf value: %f\n", node->info.leaf_value);
    else {
        printf("feature: %d, split: %f\n", node->feature_id,
               node->info.split_cond);
        for (int i = 0; i < depth; i++) printf("\t");
        printf("LEFT\n");
        printNode(node->left, depth + 1);
        for (int i = 0; i < depth; i++) printf("\t");
        printf("RIGHT\n");
        printNode(node->right, depth + 1);
    }
}
void printTree(XGBoostTree *tree) { printNode(tree->root, 0); }