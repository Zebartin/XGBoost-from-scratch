#ifndef XGB_H_
#define XGB_H_
#include "data.h"
#include "tree.h"

typedef struct {
    int n_estimator;
    int max_depth;
    double shrinkage, gamma, lambda;
    void (*calGradientAndHessian)(double *, Data *);
    XGBoostTree *trees;
} XGBoostModel;

XGBoostModel *createXGBoostModel(const char *type);

void fitModel(Data *Xy, XGBoostModel *model);

void predictModel(Data *Xy, double *outy, XGBoostModel *model);
#endif