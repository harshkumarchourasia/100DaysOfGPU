# CUDA 100 days Learning Journey



Day | Files & Summaries | Reference Material | 
:---: |:---: | :--: |
1 | parallelize_for_loop.cu: Increament each element in loop by a constant. | Josh Holloway - Intro to CUDA series.
2 | mat_mul.cu: Multiply two matrices | PMMP
3 | tiled_mat_mul.cu: Tiled matrix multiplication | PMMP
4 | conv1d.cu: Conv1D implementation | PMMP
5 | conv2Dnaive.cu: Conv2D naive implementation; conv2Doptimized.cu Optimized kernel | PMMP
6 | conv2DpinnedMem.cu: Conv2D with pinned memory 230X faster | PMMP
7 | atomic_ops.cu: Learnt about cuda atomic operations and implemented and benchmarked histogram problem | cuda_by_example
8 | reduction.cu: Basics of reduction in CUDA | PMMP
9 | max_reduction.cu: Implemented optimized max reduction algorithm [10X faster than CPU algorithm] | PMMP
10 | prefix_sum.cu: Implemented naive prefix sum algo in cuda c (TODO: Improve the algorithm) | PMMP
11 | prefix_sum.cu: Implemented optimized prefix sum algo in cuda c | PMMP
12 | pytorch-profiling.ipynb: Learnt about profiling and running CUDA code in pytorch | GPU mode
13 | Started challenges. Day0 Challenge done! | ppc-exercises.cs.aalto.fi
14 | CP 1 Done | ppc-exercises.cs.aalto.fi
15 | CP 4 Done | ppc-exercises.cs.aalto.fi
16 | CP 5 Started | ppc-exercises.cs.aalto.fi
17 | Matrix transpose implementation | NA
18 | Non square matrix multiplication | NA
19 | CP5 Optimized 25s => 5s runtime | ppc-exercises.cs.aalto.fi
20-21 | More optimizations for CP5 Runtime 7.9 sec to 0.08 sec | ppc-exercises.cs.aalto.fi
22 | SO6 | Radix sort | ppc-exercises.cs.aalto.fi