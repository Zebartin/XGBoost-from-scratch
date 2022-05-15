
#include "xgb.h"

#include <string.h>

#include "tree.h"
#include "utils.h"
void calGHForRegression(double *predy, Data *Xy) {
    for (int i = 0; i < Xy->n_example; i++) {
        Xy->gradient[i] = predy[i] - Xy->y[i];
        Xy->hessian[i] = 1;
    }
}
void calGHForClassification(double *predy, Data *Xy) {
    for (int i = 0; i < Xy->n_example; i++) {
        double t = sigmoid(predy[i]);
        Xy->gradient[i] = t - Xy->y[i];
        Xy->hessian[i] = fmax(t * (1 - t), 1e-16);
    }
}
XGBoostModel *createXGBoostModel(const char *type) {
    XGBoostModel *ret = mallocOrDie(sizeof(XGBoostModel));
    if (strcmp("regression", type) == 0)
        ret->calGradientAndHessian = &calGHForRegression;
    else if (strcmp("classification", type) == 0)
        ret->calGradientAndHessian = &calGHForClassification;
    else {
        fprintf(stderr, "Invalid model type: %s\n", type);
        exit(-1);
    }
    ret->gamma = 0;
    ret->lambda = 0;
    ret->shrinkage = 0.3;
    ret->max_depth = 2;
    ret->n_estimator = 128;
    return ret;
}
void fitModel(Data *Xy, XGBoostModel *model) {
    double *outy = mallocOrDie(sizeof(double) * Xy->n_example);
    double *t = mallocOrDie(sizeof(double) * Xy->n_example);
    model->trees = mallocOrDie(sizeof(XGBoostTree) * model->n_estimator);
    memset(outy, 0, sizeof(double) * Xy->n_example);
    for (int i = 0; i < model->n_estimator; i++) {
        model->calGradientAndHessian(outy, Xy);
        model->trees[i].gamma = model->gamma;
        model->trees[i].lambda = model->lambda;
        model->trees[i].max_depth = model->max_depth;
        fitTree(Xy, model->trees + i);
        predictTree(Xy, t, model->trees + i);
        for (int j = 0; j < Xy->n_example; j++)
            outy[j] += t[j] * model->shrinkage;
        // printTree(model->trees + i);
        // for (int k = 0; k < 10; k++)
        //     printf("%d\t%f\t%f\n", k, t[k], outy[k]);
        // printf("\n");
    }
    free(outy);
    free(t);
}

void predictModel(Data *Xy, double *outy, XGBoostModel *model) {
    double *t = mallocOrDie(sizeof(double) * Xy->n_example);
    memset(outy, 0, sizeof(double) * Xy->n_example);
    for (int i = 0; i < model->n_estimator; i++) {
        predictTree(Xy, t, model->trees + i);
        for (int j = 0; j < Xy->n_example; j++)
            outy[j] += t[j] * model->shrinkage;
    }
}