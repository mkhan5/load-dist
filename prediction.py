import sys
import math

# Formula to calculate execution time
size_of_mat = 2000
size_of_test = 1000
# server_1 time on CPU and GPU
time_test_gpu = 0.0384   # 0.014
time_test_cpu = 0.0404  # 0.0035 for 24 threads & 0.033 for 16 threads

# server_2 time on CPU and GPU
#
# time_test_gpu = 0.006
# time_test_cpu = 0.025 # 0.0210 for 16 threads & 0.0207 for 24 threads



for size in range(1000,10000,1000):
    ratio = size/size_of_test
    powg = 2  # math.log(size)
    powc = 2.9
    gpu_pred_time = pow(ratio,powg)*time_test_gpu
    cpu_pred_time = pow(ratio,powc)*time_test_cpu

    print "--------------------------------------"
    print "Pred time for",size,"K on GPU =",round(gpu_pred_time,3),"s"
    print "Pred time for",size,"K on CPU =",round(cpu_pred_time,3),"s"

    scale = 0.05
    pred_scale = 1
    min_time = 0
    min_cpu_t = 0
    min_gpu_t = 0
    least_exec_time = 9998.121
    for sc in range(0,19):
        ratio2a = (scale*size)/size_of_test
        ratio2b = ((1-scale)*size)/size_of_test
        gpu_pred_time2 = pow(ratio2a,powg)*time_test_gpu
        cpu_pred_time2 = pow(ratio2b,powc)*time_test_cpu
        cpu_gpu_pred_time = max( gpu_pred_time2, cpu_pred_time2)
        min_time = cpu_gpu_pred_time
        # print "---Pred time for",size,"K on GPU =",round(gpu_pred_time2,3),"s"
        # print "---Pred time for",size,"K on CPU =",round(cpu_pred_time2,3),"s"
        # print "---Pred time for",size,"K on CPU+GPU =",cpu_gpu_pred_time,"s for scale",scale
        if min_time <= least_exec_time:
            least_exec_time = min_time
            pred_scale = scale
            min_cpu_t = cpu_pred_time2
	    min_gpu_t = gpu_pred_time2
        scale+=0.05
    print "Pred time for",size,"K on CPU+GPU =",round(least_exec_time,3),"s for scale",pred_scale
    print " [ratio on gpu ",pred_scale*size,"pred time gpu ",round(min_gpu_t,3),"pred time cpu ",round(min_cpu_t,3) , "]"
