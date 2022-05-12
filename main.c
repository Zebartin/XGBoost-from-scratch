#include <stdio.h>
#include <stdlib.h>

#include "data.h"
#include "tree.h"
#include "utils.h"
#include "xgb.h"

int main() {
    Data *Xy = readCSV("cate_mushrooms.csv", ",", 0);
    XGBoostModel *m = createXGBoostModel("classification");
    m->gamma = 0;
    m->lambda = 0;
    m->max_depth = 3;
    m->shrinkage = 0.5;
    m->n_estimator = 30;
    fitModel(Xy, m);
    double *outy = mallocOrDie(sizeof(double) * Xy->n_example);
    predictModel(Xy, outy, m);
    int a = 0, b = 0, c = 0;
    for (int i = 0; i < Xy->n_example; i++) {
        int py = sigmoid(outy[i]) > 0.5;
        int ry = (int)(Xy->y[i]);
        if (ry == 1) {
            if (py == 1)
                a++;
            else
                b++;
        } else if (py == 1)
            c++;
    }
    printf("全1：%d, 全0：%d, 实1预0：%d, 实0预1：%d\n", a,
           Xy->n_example - a - b - c, b, c);
    // int N = 30;
    // Data *Xy = malloc(sizeof(Data));
    // double *outy = malloc(sizeof(double) * N);
    // Xy->n_example = N;
    // Xy->n_feature = 1;
    // Xy->X = malloc(sizeof(double *) * N);
    // Xy->y = malloc(sizeof(double) * N);
    // Xy->gradient = malloc(sizeof(double) * N);
    // Xy->hessian = malloc(sizeof(double) * N);
    // Xy->feature_blocks = (int **)malloc(sizeof(int *));
    // Xy->feature_blocks[0] = (int *)malloc(sizeof(int) * N);
    // for (int i = 0; i < N; i++) {
    //     Xy->X[i] = malloc(sizeof(double));
    //     Xy->X[i][0] = i;
    //     Xy->y[i] = rand() % 2;
    //     Xy->feature_blocks[0][i] = i;
    // }
    // XGBoostModel *m = createXGBoostModel("classification");
    // m->gamma = 0;
    // m->lambda = 0;
    // m->max_depth = 2;
    // m->n_estimator = 30;
    // m->shrinkage = 1;
    // fitModel(Xy, m);
    // predictModel(Xy, outy, m);
    // for (int i = 0; i < N; i++) printf("%d\t%f\t%f\n", i, sigmoid(outy[i]),
    // Xy->y[i]);
}
