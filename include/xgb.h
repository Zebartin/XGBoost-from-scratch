#ifndef XGB_H_
#define XGB_H_
#include "data.h"
#include "tree.h"

enum model_type{Regression, BinaryClassification, MultiClassification};
typedef struct {
    enum model_type mtype;
    int n_group;                    // 用于多分类
    int n_estimator;                // 树的数量
    int max_depth;
    double shrinkage, gamma, lambda;
    void (*calGradientAndHessian)(double *, Data *, GradientPair *);
    XGBoostTree *trees;
} XGBoostModel;

XGBoostModel *createXGBoostModel(enum model_type type);

void fitModel(Data *Xy, XGBoostModel *model);

void predictModel(Data *Xy, double *outy, XGBoostModel *model);
#endif