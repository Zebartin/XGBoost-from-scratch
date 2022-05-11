#include "data.h"

int comp(const void *a, const void *b) { return (*(int *)a - *(int *)b); }

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
