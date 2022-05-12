#ifndef UTILS_H_
#define UTILS_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static inline void *mallocOrDie(size_t MemSize) {
    void *AllocMem = malloc(MemSize);
    if (!AllocMem && MemSize) {
        fprintf(stderr, "failed to malloc\n");
        exit(-1);
    }
    return AllocMem;
}

static inline double sigmoid(double x) { return 1 / (1 + exp(-x)); }
#endif