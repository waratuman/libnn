// The derivative paramter is the nth derivative of the function:
// - 0 is the function
// - 1 is the first derivative
// - 2 is the 2nd derivative
// - -1 is the first integral
// - -2 is the sencond integral
// etc..
typedef float (*nn_error_fn)(float a, float b, int derivative);
typedef float (*nn_error_integration_fn)(int n, float* a);

// f(a, b) = 1 / 2 (b - a) ^ 2
// f'(a, b) = (b - a)
float squared_error(float a, float b, int d);

float sum_error_integration(int n, float* a);
