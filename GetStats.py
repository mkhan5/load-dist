import subprocess
import os
"""
File used to automatically compile, run and time the Polybench executables on
 a given platform and compiler

"""

kernels = {
    "blas": ["gemm", "gemver", "gesummv", "symm", "syr2k", "syrk", "trmm"],
    "kernels": ["2mm", "3mm", "atax", "bicg", "mvt"],
    "solvers": ["cholesky", "gramschmidt", "lu", "ludcmp", "trisolv"]
}
server = "IITB"

def get_compile_command(compiler_used, type, kernel, platform = ""):
    compiler = compiler_used
    include_flags = " -w -I utilities -I linear-algebra/" + type + "/" + kernel
    c_files = " utilities/polybench.c linear-algebra/" + type + "/" + kernel + "/" + kernel + ".c"
    other_flags = " -O3 -lm -DEXTRALARGE_DATASET -DPOLYBENCH_TIME -o bin/" + kernel + "_" + compiler
    extension = ""
    mkl_includes = ""
    mkl_lib = ""
    cuda_includes = ""
    cuda_lib = ""
    if platform == "openmp":
        extension = ".c"
        if server == "IITB":
            mkl_includes = " -DMKL_ILP64 -fopenmp -m64 -I/home/gpu_users/ahmedw/Intelmkl/Intel/compilers_and_libraries_2016.1.150/linux/mkl/include "
            mkl_lib = " -L/home/gpu_users/ahmedw/Intelmkl/Intel/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64_lin -lmkl_intel_ilp64 -lmkl_core -lmkl_gnu_thread -lm "
        else:
            mkl_includes = " -DMKL_ILP64 -fopenmp -m64 -I/opt/intel/compilers_and_libraries_2016.1.150/linux/mkl/include "
            mkl_lib = " -L/opt/intel/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64_lin -lmkl_intel_ilp64 -lmkl_core -lmkl_gnu_thread -lm "
    elif platform == "cuda":
        extension = ".c.c"
        cuda_includes = " -Icommon/inc -I/usr/local/cuda-7.0/include/ "
        cuda_lib = " -L/usr/local/cuda-7.0/lib64 -L/usr/local/cuda-7.0/lib -lm -lcudart  -lcublas -lcusolver "

    other_flags += extension

    command = compiler + include_flags + mkl_includes + cuda_includes + c_files + extension + other_flags + mkl_lib  + cuda_lib
    executable_name = "./bin/" + kernel + "_" + compiler + extension
    return command, executable_name

def get_results(command, exec_name):
    result = subprocess.Popen(command, shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result.wait()
    timing_list = []
    if result.returncode == 0:
        for i in range(0, 5):
            tmpres = subprocess.Popen(exec_name, shell=True, cwd=os.getcwd(), stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
            tmpres.wait()
            output = tmpres.stdout.read().strip()
            timing_list.append(float(output))
        return float(sum(timing_list)) / max(len(timing_list), 1)
    else:
        return 0
#prev_dir = os.getcwd()
#os.chdir("../")
#working_dir = os.getcwd() + "/input/polybench-c-4.2.1-beta"
#os.chdir(working_dir)

# print "type_kernel\t time_with_gcc\t time_with_openmp\t speedup"
print "type \t kernel\t time_with_gcc\t time_with_mkl\t time_with_cublas\t speedup with mkl\t speedup with cublas"
for type in kernels:
    for kernel in kernels[type]:
        command1, exec_name1 = get_compile_command("gcc", type, kernel)
        gcc_timings = get_results(command1, exec_name1)
        command2, exec_name2 = get_compile_command("gcc", type, kernel, platform="openmp")
        mkl_timings = get_results(command2, exec_name2)
        command3, exec_name3 = get_compile_command("nvcc", type, kernel, platform="cuda")
        #print command3, exec_name3
        cublas_timings = get_results(command3, exec_name3)
        # print type, "\t",kernel, "\t", gcc_timings, "\t", mkl_timings, "\t", gcc_timings/mkl_timings
        print type, "\t", kernel, "\t", round(gcc_timings,6), "\t", round(mkl_timings,6), "\t", round(cublas_timings,6), "\t", round(gcc_timings / mkl_timings,6), "\t", round(gcc_timings / cublas_timings,6)

#os.chdir(prev_dir)
        # TODO
        # 1. Compile the file using the 'command' obtained
        # 2. Run 'exec_name' obtained from above 5 times using a system call
        # 3. Capture the output (timing) to a variable each time
        # 4. Average the timing and display in the following format (4 values) for each kernel
        #       type_kernel     time_with gcc   time_with_openmp    speedup
        #

