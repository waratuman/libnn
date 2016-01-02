// The weights are passed in as a
// Input passed in as b
typedef float (*nn_integration_fn)(int count, float* weights, float* input);

// Sum of Products
// \sum_{i=0}^{n} \left( a_i b_i \right)
float sum_of_products_integration(int n, float* a, float* b);

// Max (Usefull for max pooling in a convolutional network)
// max(b)
float max_integration(int n, float* a, float* b);
