nvcc -arch sm_60 -v --ptxas-options=-v --use_fast_math -lcufft -O3 seconds.cpp main.cu -o sim