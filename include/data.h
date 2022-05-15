#ifndef DATA_H_
#define DATA_H_

typedef struct {
    double g,h;
} GradientPair;

typedef struct {
    int n_example, n_feature, n_group;
    double **X, *y;
    int **feature_blocks;
} Data;

typedef struct {
    int *indices;
    int size;
} Subset;

int comp_int(const void *a, const void *b);

int inSubset(int x, Subset *subset);
Subset *initSubset(int *indices, int size);

Data *readCSV(const char *file_path, const char *delimiter, int firstLineIsHeader);
#endif