#pragma once

/* = Kernel Functions ==========================================================
 * TODO: Should not be in public interface
 * The kernel offset is the difference between any element and the center of the
 * kernel. This is usefull for calculating the input into the kernel given the
 * center input index of the kernel.
 *
 * For example. A 3x3 kernel would return:
 * [ [ -1, -1 ], [ -1, 0 ], [ -1, 1 ],
 *   [  0, -1 ], [  0, 0 ], [  0, 1 ],
 *    [ 1, -1 ], [  1, 0 ], [  1, 1 ] ]
 * 
 * The result is stored into the result argument, which must have allocated enough
 * space for all the results, which is the product of each dimesion (3x3 in the case
 * of the example).
 * 
 * Arguments:
 * dimensionCount: The number of the dimensions of the kernel
 *     dimensions: The dimensions of the kernel
 *         result: A pointer to an array of arrays of ints in which to store the result
 */
void nn_kernel_offset(int dimensionCount, int* dimensions, int** result);

void nn_kernel_center(int dimesionCount, int* dimesions, int* result);


// = Utitlity Functions ========================================================

/* Calculates the dot product of the vectors a and b (a \cdotp b)
 *
 *  n: The number of elements in the vectors a, b
 *  a: Vector a
 * as: Stride within vector a, if as is 7 every 7th element is used
 *  b: Vector b
 * bs: Stride within vector b, if bs is 7 every 7th element is used
 */
float nn_sdot(int n, float* a, int as, float* b, int bs);

/* Calculates the hadamard product of the vectors a and b. The result is stored
 * in the vector c (c = \alpha a \circ b).
 * 
 *     n: The number of elements in the vectors a, b
 * alpha: Scale the result by
 *     a: Vector a
 *    as: Stride within vector a, if as is 7 every 7th element is used
 *  beta: Scale vector b
 *     b: Vector b
 *    bs: Stride within vector b, if bs is 7 every 7th element is used
 *     c: Vector c
 *    cs: Stride within vector b, if cs is 7 every 7th element is used
 */
void nn_shad(int n, float alpha, float* a, int as, float* b, int bs, float* c, int cs);

/* Converts an 1D input index to an N dimensional index */
void nn_ii2di(int dimensions, int* dimensionality, int ii, int* result);

/* Convert an N dimensional index to a 1D input index */
int nn_di2ii(int dimensions, int* dimensionality, int* di);

// Calculates the addition of two vectors a and b where a is scaled by alpha
// and stores the reult in c (c = \alpha a + b).
//     n: The number of elements in the vectors a, b
// alpha: Scale of vector a
//     a: Vector a
//    as: Stride within vector a, if as is 7 every 7th element is used
//     b: Vector b
//    bs: Stride within vector b
// The result is stored in vector b;
// void nn_saxpy(int n, float alpha, float* a, int as, float* b, int bs);
