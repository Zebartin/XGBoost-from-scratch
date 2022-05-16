
#include "xgb.h"

#include <string.h>

#include "tree.h"
#include "utils.h"
// 参考：https://github.com/dmlc/xgboost/blob/4224c08cacceba3f83f90e387c07aa6205a83bfa/src/objective/regression_loss.h#L17
void calGHForRegression(double *predy, Data *Xy, GradientPair *gpair) {
    for (int i = 0; i < Xy->n_example; i++) {
        gpair[i].g = predy[i] - Xy->y[i];
        gpair[i].h = 1;
    }
}
// 参考：https://github.com/dmlc/xgboost/blob/4224c08cacceba3f83f90e387c07aa6205a83bfa/src/objective/regression_loss.h#L67
void calGHForBinary(double *predy, Data *Xy, GradientPair *gpair) {
    for (int i = 0; i < Xy->n_example; i++) {
        double t = sigmoid(predy[i]);
        gpair[i].g = t - Xy->y[i];
        gpair[i].h = fmax(t * (1 - t), 1e-16);
    }
}
// 参考：https://github.com/dmlc/xgboost/blob/4224c08cacceba3f83f90e387c07aa6205a83bfa/src/objective/multiclass_obj.cu#L96
void calGHForMulti(double *predy, Data *Xy, GradientPair *gpair) {
    int n_group = Xy->n_group;
    for (int i = 0; i < Xy->n_example; i++) {
        double wmax = predy[i * n_group];
        for (int j = 1; j < n_group; j++) {
            wmax = fmax(wmax, predy[i * n_group + j]);
        }
        double wsum = 0;
        for (int j = 0; j < n_group; j++) {
            wsum += exp(predy[i * n_group + j] - wmax);
        }
        int label = Xy->y[i];
        for (int j = 0; j < n_group; j++) {
            double p = exp(predy[i * n_group + j] - wmax) / wsum;
            gpair[i * n_group + j].g = label == j ? p - 1.0 : p;
            gpair[i * n_group + j].h = fmax(2.0 * p * (1.0 - p), 1e-16);
        }
    }
}
XGBoostModel *createXGBoostModel(enum model_type type) {
    XGBoostModel *ret = mallocOrDie(sizeof(XGBoostModel));
    if (type == Regression)
        ret->calGradientAndHessian = &calGHForRegression;
    else if (type == BinaryClassification)
        ret->calGradientAndHessian = &calGHForBinary;
    else if (type == MultiClassification)
        ret->calGradientAndHessian = &calGHForMulti;
    else {
        fprintf(stderr, "Invalid model type\n");
        exit(-1);
    }
    ret->mtype = type;
    ret->n_group = 1;
    ret->gamma = 0;
    ret->lambda = 1;
    ret->shrinkage = 0.3;
    ret->max_depth = 2;
    ret->n_estimator = 64;
    return ret;
}
void fitModel(Data *Xy, XGBoostModel *model) {
    if (model->mtype == MultiClassification){
        // 统计y的种类数量，输入须保证y的取值形如[0,1,...]
        for (int i = 0; i < Xy->n_example; i++) {
            int ty = (int)(Xy->y[i]);
            if (ty > Xy->n_group)
                Xy->n_group = ty;
        }
        model->n_group = ++Xy->n_group;
    }
    int n_group = model->n_group;
    double *outy = mallocOrDie(sizeof(double) * Xy->n_example * n_group);
    GradientPair *gpair =
        mallocOrDie(sizeof(GradientPair) * Xy->n_example * n_group);
    double *t = mallocOrDie(sizeof(double) * Xy->n_example);
    model->trees =
        mallocOrDie(sizeof(XGBoostTree) * model->n_estimator * n_group);
    memset(outy, 0, sizeof(double) * Xy->n_example * n_group);
    if (model->mtype != MultiClassification) {
        for (int i = 0; i < model->n_estimator; i++) {
            model->calGradientAndHessian(outy, Xy, gpair);
            model->trees[i].gamma = model->gamma;
            model->trees[i].lambda = model->lambda;
            model->trees[i].max_depth = model->max_depth;
            fitTree(Xy, gpair, model->trees + i);
            predictTree(Xy, t, model->trees + i);
            for (int j = 0; j < Xy->n_example; j++)
                outy[j] += t[j] * model->shrinkage;
        }
    } else {
        GradientPair *tmp_gpair =
            mallocOrDie(sizeof(GradientPair) * Xy->n_example);
        for (int i = 0; i < model->n_estimator; i++) {
            model->calGradientAndHessian(outy, Xy, gpair);
            for (int j = 0; j < n_group; j++) {
                for (int k = 0; k < Xy->n_example; k++) {
                    tmp_gpair[k] = gpair[k * n_group + j];
                }
                int idx = i * n_group + j;
                model->trees[idx].gamma = model->gamma;
                model->trees[idx].lambda = model->lambda;
                model->trees[idx].max_depth = model->max_depth;
                fitTree(Xy, tmp_gpair, model->trees + idx);
                predictTree(Xy, t, model->trees + idx);
                for (int k = 0; k < Xy->n_example; k++)
                    outy[k * n_group + j] += t[k] * model->shrinkage;
            }
        }
        free(tmp_gpair);
    }
    free(outy);
    free(t);
    free(gpair);
}

void predictModel(Data *Xy, double *outy, XGBoostModel *model) {
    int n_group = model->n_group;
    double *t = mallocOrDie(sizeof(double) * Xy->n_example);
    memset(outy, 0, sizeof(double) * Xy->n_example);
    if (model->mtype != MultiClassification) {
        for (int i = 0; i < model->n_estimator; i++) {
            predictTree(Xy, t, model->trees + i);
            for (int j = 0; j < Xy->n_example; j++)
                outy[j] += t[j] * model->shrinkage;
        }
        if (model->mtype == BinaryClassification)
            for (int i = 0; i < Xy->n_example; i++)
                outy[i] = sigmoid(outy[i]) >= 0.5;
    } else {
        double *tout = mallocOrDie(sizeof(double) * Xy->n_example * n_group);
        memset(tout, 0, sizeof(double) * Xy->n_example * n_group);
        for (int i = 0; i < model->n_estimator; i++) {
            for (int j = 0; j < model->n_group; j++) {
                predictTree(Xy, t, model->trees + i * n_group + j);
                for (int k = 0; k < Xy->n_example; k++)
                    tout[k * n_group + j] += t[k] * model->shrinkage;
            }
        }
        for (int i = 0; i < Xy->n_example; i++) {
            double wmax = tout[i * n_group];
            int pred = 0;
            for (int j = 1; j < n_group; j++)
                if (tout[i * n_group + j] > wmax) {
                    pred = j;
                    wmax = tout[i * n_group + j];
                }
            outy[i] = pred;
        }
        free(tout);
    }
    free(t);
}