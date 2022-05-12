#include "data.h"

#include <stdio.h>

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
// from https://www.geeksforgeeks.org/binary-search/
int inSubset(int x, Subset *subset) {
    int l = 0, r = subset->size - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;

        // Check if x is present at mid
        if (subset->indices[m] == x)
            return 1;

        // If x greater, ignore left half
        if (subset->indices[m] < x)
            l = m + 1;

        // If x is smaller, ignore right half
        else
            r = m - 1;
    }

    // if we reach here, then element was
    // not present
    return 0;
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
    data->X = mallocOrDie(sizeof(double *) * i);
    data->y = mallocOrDie(sizeof(double) * i);
    data->gradient = mallocOrDie(sizeof(double) * i);
    data->hessian = mallocOrDie(sizeof(double) * i);
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