#ifndef GEN_CUDA_CLIENT_H
#define GEN_CUDA_CLIENT_H

void *get_function_pointer(const char *function_name);
void *get_per_thread_function(void *function);
#endif
