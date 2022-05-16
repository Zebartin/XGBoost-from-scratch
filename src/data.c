#include "data.h"

#include <stdio.h>
#include <string.h>

#include "csvparser.h"
#include "utils.h"

int comp_int(const void *a, const void *b) { return (*(int *)a - *(int *)b); }

typedef struct {
    int index;
    double val;
} ivpair;

int comp_pair(const void *a, const void *b) {
    ivpair *pa = (ivpair *)a, *pb = (ivpair *)b;
    return pa->val - pb->val;
}

Subset *initSubset(int size, int init_val) {
    Subset *ret = mallocOrDie(sizeof(Subset));
    ret->size = size;
    ret->bitset = mallocOrDie(sizeof(char) * BITNSLOTS(size));
    if (init_val) {
        memset(ret->bitset, -1, sizeof(char) * BITNSLOTS(size));
        ret->cnt = size;
    } else {
        memset(ret->bitset, 0, sizeof(char) * BITNSLOTS(size));
        ret->cnt = 0;
    }
    return ret;
}

void freeSubset(Subset *subset) {
    free(subset->bitset);
    free(subset);
}

int inSubset(int x, Subset *subset) {
    if (x < 0 || x >= subset->size)
        return 0;
    return BITTEST(subset->bitset, x);
}

void addToSubset(int x, Subset *subset) {
    if (x < 0 || x >= subset->size)
        return;
    if (BITTEST(subset->bitset, x))
        return;
    BITSET(subset->bitset, x);
    subset->cnt++;
}

void resetSubset(Subset *subset){
    memset(subset->bitset, 0, sizeof(char)*BITNSLOTS(subset->size));
    subset->cnt=0;
}
Data *readCSV(const char *file_path, const char *delimiter,
              int first_line_is_header) {
    FILE *fp = fopen(file_path, "r");
    // 统计行数
    int i = 0;
    while (EOF != (fscanf(fp, "%*[^\n]"), fscanf(fp, "%*c"))) i++;
    fclose(fp);
    Data *data = mallocOrDie(sizeof(Data));
    data->n_example = i;
    data->n_feature = -1;
    data->n_group = 1;
    data->X = mallocOrDie(sizeof(double *) * i);
    data->y = mallocOrDie(sizeof(double) * i);
    // 逐行读取
    CsvParser *csvparser =
        CsvParser_new(file_path, delimiter, first_line_is_header);
    CsvRow *row;
    i = 0;
    while ((row = CsvParser_getRow(csvparser))) {
        const char **rowFields = CsvParser_getFields(row);
        if (data->n_feature == -1)
            data->n_feature = CsvParser_getNumFields(row) - 1;
        data->X[i] = mallocOrDie(sizeof(double) * data->n_feature);
        for (int j = 0; j < data->n_feature; j++)
            sscanf(rowFields[j + 1], "%lf", data->X[i] + j);
        sscanf(rowFields[0], "%lf", data->y + i);
        i++;
        CsvParser_destroy_row(row);
    }
    CsvParser_destroy(csvparser);
    // 计算排好序的feature_block
    data->feature_blocks = mallocOrDie(data->n_feature * sizeof(int *));
    ivpair *t = mallocOrDie(data->n_example * sizeof(ivpair));
    for (int j = 0; j < data->n_feature; j++) {
        for (i = 0; i < data->n_example; i++) {
            t[i].index = i;
            t[i].val = data->X[i][j];
        }
        qsort(t, data->n_example, sizeof(ivpair), comp_pair);
        data->feature_blocks[j] = mallocOrDie(data->n_example * sizeof(int));
        for (i = 0; i < data->n_example; i++)
            data->feature_blocks[j][i] = t[i].index;
    }
    free(t);
    return data;
}

void printConfusionMatrix(Data *Xy, double *outy) {
    int n_group = Xy->n_group;
    if (n_group == 1)
        n_group = 2;
    int **conm = mallocOrDie(sizeof(int *) * n_group);
    for (int i = 0; i < n_group; i++) {
        conm[i] = mallocOrDie(sizeof(int) * n_group);
        memset(conm[i], 0, sizeof(int) * n_group);
    }
    int realy, predy;
    for (int i = 0; i < Xy->n_example; i++) {
        realy = (int)(Xy->y[i]);
        predy = (int)outy[i];
        conm[realy][predy]++;
    }
    for (int i = 0; i < n_group; i++) {
        printf("\t%d", i);
    }
    printf("\trecall\n");
    for (int i = 0; i < n_group; i++) {
        int true_positive = conm[i][i], total = 0;
        printf("%d", i);
        for (int j = 0; j < n_group; j++) {
            printf("\t%d", conm[i][j]);
            total += conm[i][j];
        }
        printf("\t%.2f%%\n", true_positive * 100.0 / total);
    }
    printf("prec");
    int all_true = 0;
    for (int j = 0; j < n_group; j++) {
        int true_positive = conm[j][j], total = 0;
        for (int i = 0; i < n_group; i++) {
            total += conm[i][j];
        }
        all_true += conm[j][j];
        printf("\t%.2f%%", true_positive * 100.0 / total);
    }
    printf("\n");
    printf("accuracy: %.2f%%\n", all_true * 100.0 / Xy->n_example);
    for (int i = 0; i < n_group; i++) free(conm[i]);
    free(conm);
}