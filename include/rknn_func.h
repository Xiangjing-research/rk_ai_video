#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define _BASETSD_H

#include "RgaUtils.h"

#include "postprocess.h"

#include "rknn_api.h"
#include "preprocess.h"

#define PERF_WITH_POST 1

void dump_tensor_attr(rknn_tensor_attr *attr);
double __get_us(struct timeval t);
unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
unsigned char *load_model(const char *filename, int *model_size);
int saveFloat(const char *file_name, float *output, int element_size);