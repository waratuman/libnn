#pragma once

// TODO: Rename to aggragation (integration is may used for the opposite of derivative)
// count is the number of elements in each arg
// args is an array of float arrays, and the number accessed is dependent on the function (most will have 2, the first beign the weights, and the second being the values)
typedef float (*nn_integration_fn)(int count, float** args);

// Sum of Products
// 2 args
// \sum_{i=0}^n \left( a_{0,i} a_{1,i} \right)
float sum_of_products_integration(int n, float** args);

// Euclidean 
// \frac{1}{2n} \sum_{i=0}^{n} \left( a_{0,i} - a_{1,i} \right)^{2}
float euclidean_integration(int n, float** args);

// Sum of Squares
// \sum_{i=0}^n \left( a_{0,i}^{2} \right)
float sum_of_squares(int n, float** args);

// Max (Usefull for max pooling in a convolutional network)
// \max\left(a_0\right)
float max_integration(int n, float** args);
