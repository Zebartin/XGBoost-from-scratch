#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "tree.h"
#include "xgb.h"

int main() {
    int N = 30;
    Data *Xy = malloc(sizeof(Data));
    double *outy = malloc(sizeof(double) * N);
    Xy->n_example = N;
    Xy->n_feature = 1;
    Xy->X = malloc(sizeof(double *) * N);
    Xy->y = malloc(sizeof(double) * N);
    Xy->gradient = malloc(sizeof(double) * N);
    Xy->hessian = malloc(sizeof(double) * N);
    Xy->feature_blocks = (int **)malloc(sizeof(int *));
    Xy->feature_blocks[0] = (int *)malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) {
        Xy->X[i] = malloc(sizeof(double));
        Xy->X[i][0] = i;
        Xy->y[i] = rand() % 2;
        Xy->feature_blocks[0][i] = i;
    }
    XGBoostModel *m = createXGBoostModel("classification");
    m->gamma = 0;
    m->lambda = 0;
    m->max_depth = 2;
    m->n_estimator = 30;
    m->shrinkage = 1;
    fitModel(Xy, m);
    predictModel(Xy, outy, m);
    for (int i = 0; i < N; i++) printf("%d\t%f\t%f\n", i, sigmoid(outy[i]), Xy->y[i]);
}
