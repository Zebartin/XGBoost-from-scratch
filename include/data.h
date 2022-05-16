#ifndef DATA_H_
#define DATA_H_

#include <limits.h> /* for CHAR_BIT */

#define BITMASK(b) (1 << ((b) % CHAR_BIT))
#define BITSLOT(b) ((b) / CHAR_BIT)
#define BITSET(a, b) ((a)[BITSLOT(b)] |= BITMASK(b))
#define BITCLEAR(a, b) ((a)[BITSLOT(b)] &= ~BITMASK(b))
#define BITTEST(a, b) ((a)[BITSLOT(b)] & BITMASK(b))
#define BITNSLOTS(nb) ((nb + CHAR_BIT - 1) / CHAR_BIT)

typedef struct {
    double g, h;
} GradientPair;

typedef struct {
    int n_example, n_feature, n_group;
    double **X, *y;
    int **feature_blocks;
} Data;

typedef struct {
    char *bitset;
    int size;
    int cnt;
} Subset;

int comp_int(const void *a, const void *b);

Subset *initSubset(int size, int init_val);

void freeSubset(Subset *subset);

int inSubset(int x, Subset *subset);

void addToSubset(int x, Subset *subset);

void resetSubset(Subset *subset);

Data *readCSV(const char *file_path, const char *delimiter,
              int firstLineIsHeader);

void printConfusionMatrix(Data *, double *);

#endif