#ifndef DATA_H_
#define DATA_H_

typedef struct {
    int n_example, n_feature;
    double **X, *y, *gradient, *hessian;
    int **feature_blocks;
} Data;

typedef struct {
    int *indices;
    int size;
} Subset;

int comp(const void *a, const void *b);

int inSubset(int x, Subset *subset);
Subset *initSubset(int *indices, int size);

#endif