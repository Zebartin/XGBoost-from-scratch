#include "tree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#define score(G, H) (((G) * (G)) / ((H) + tree->lambda))
#define weight(G, H) (-(G) / ((H) + tree->lambda))
#define isLeaf(node) ((node)->left == NULL)

TreeNode *splitNode(Data *Xy, Subset *index_subset, int cur_depth,
                    XGBoostTree *tree) {
    double best_score = 0, before_score, after_score;
    double G = 0, H = 0;
    int best_feature, best_left_cnt = 0;
    TreeNode *ret = mallocOrDie(sizeof(TreeNode));
    for (int i = 0; i < Xy->n_example; i++)
        if (inSubset(i, index_subset)) {
            G += Xy->gradient[i];
            H += Xy->hessian[i];
        }
    if (cur_depth == tree->max_depth) {
        ret->left = NULL;
        ret->right = NULL;
        ret->info.leaf_value = weight(G, H);
        return ret;
    }
    int *left_indices = mallocOrDie(sizeof(int) * index_subset->size);
    int *best_left_indices = mallocOrDie(sizeof(int) * index_subset->size);
    before_score = score(G, H);
    for (int i = 0; i < Xy->n_feature; i++) {
        int left_cnt = 0;
        double G_L = 0, H_L = 0, G_R, H_R;
        for (int j = 0; j < Xy->n_example; j++) {
            int example_index = Xy->feature_blocks[i][j];
            if (!inSubset(example_index, index_subset))
                continue;
            left_indices[left_cnt++] = example_index;
            G_L += Xy->gradient[example_index];
            H_L += Xy->hessian[example_index];
            G_R = G - G_L;
            H_R = H - H_L;
            after_score = score(G_L, H_L) + score(G_R, H_R);
            if (after_score > best_score) {
                if (i == best_feature)
                    memcpy(best_left_indices + best_left_cnt,
                           left_indices + best_left_cnt,
                           sizeof(int) * (left_cnt - best_left_cnt));
                else
                    memcpy(best_left_indices, left_indices,
                           sizeof(int) * left_cnt);
                best_feature = i;
                best_left_cnt = left_cnt;
                best_score = after_score;
            }
        }
    }
    if (best_score > before_score + tree->gamma) {
        ret->feature_id = best_feature;
        ret->info.split_cond =
            Xy->feature_blocks[best_feature]
                              [best_left_indices[best_left_cnt - 1]];
        qsort(best_left_indices, best_left_cnt, sizeof(int), comp);
        Subset *next_subset = mallocOrDie(sizeof(Subset));
        next_subset->size = best_left_cnt;
        next_subset->indices = best_left_indices;
        ret->left = splitNode(Xy, next_subset, cur_depth + 1, tree);

        next_subset->size = index_subset->size - best_left_cnt;
        next_subset->indices = mallocOrDie(sizeof(int) * (next_subset->size));
        for (int i = 0, j = 0, k = 0; i < index_subset->size; i++) {
            if (index_subset->indices[i] == best_left_indices[j]) {
                j++;
                continue;
            }
            next_subset->indices[k++] = index_subset->indices[i];
        }
        ret->right = splitNode(Xy, next_subset, cur_depth + 1, tree);
        free(next_subset->indices);
        free(next_subset);
    } else {
        ret->left = NULL;
        ret->right = NULL;
        ret->info.leaf_value = weight(G, H);
    }
    free(left_indices);
    free(best_left_indices);
    return ret;
}

void fitTree(Data *Xy, XGBoostTree *tree) {
    Subset ss;
    ss.size = Xy->n_example;
    ss.indices = mallocOrDie(sizeof(int) * ss.size);
    for (int i = 0; i < ss.size; i++) ss.indices[i] = i;
    tree->root = splitNode(Xy, &ss, 0, tree);
    free(ss.indices);
    ss.indices = NULL;
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