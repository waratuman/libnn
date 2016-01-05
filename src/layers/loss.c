#include <stdlib.h>

#include "loss.h"
#include "math.h"
#include "integrations.h"

void nn_layer_init_loss(nn_layer_loss_t *layer)
{

}

nn_layer_loss_t* nn_layer_create_loss(nn_integration_fn fl, int n)
{
    nn_layer_loss_t* l = calloc(1, sizeof(nn_layer_loss_t));
    l->fn = fl;
    l->inputCount = n;

    nn_layer_init_loss(l);
    return l;
}

void nn_layer_destroy_loss(nn_layer_loss_t* l)
{
    free(l);
}

float nn_layer_activate_loss(nn_layer_loss_t* l, float* input, float* output)
{
    return l->fn(l->inputCount, input, output);
}
