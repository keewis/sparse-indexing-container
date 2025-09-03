# Accelerated indexing and concatenation for sparse arrays

If we can avoid checking the coord and data arrays on the construction of subarrays, slicing can become much quicker. For example, slicing a sparse array with `n` non-background values becomes at worst `O(n)` for COO, and can possibly be accelerated even more if we build a tree for the coordinates.
