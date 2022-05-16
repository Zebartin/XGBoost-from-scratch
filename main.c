#include <stdio.h>
#include <stdlib.h>

#include "data.h"
#include "tree.h"
#include "utils.h"
#include "xgb.h"

int main() {
    Data *Xy = readCSV("example-dataset/cate_mushrooms.csv", ",", 0);
    XGBoostModel *m = createXGBoostModel(MultiClassification);
    fitModel(Xy, m);
    double *outy = mallocOrDie(sizeof(double) * Xy->n_example);
    predictModel(Xy, outy, m);
    printConfusionMatrix(Xy, outy);
}
