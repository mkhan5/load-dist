import os
import sys
import subprocess
import xlwt
sys.path.extend(['.', '..'])

# TODO
# Find the least exec time and give the parameters for it
# Write temp data to a text file so that the excel file can be cross checked

cmd1 = " ./simpleCUBLAS"
pardir = os.path.normpath(os.getcwd() + os.sep + os.pardir)

scale = 0

f = open("output_info_cublas.txt",'w')
wb = xlwt.Workbook()
ws = wb.add_sheet("My Result Sheet")
ws.write(0,0,"matrix size") # row-dynamic, col1-threads, col2-scale, col3-exec_time
ws.write(0,1,"threads")
ws.write(0,2,"scale")
ws.write(0,3,"exec time in sec")

i=1
k=20
least_exec_time = 9998.121
pars_least_exec_time =[]
tmpres = r"""MKL (%2.3f) + CUBLAS (%2.3f) - Total Time for matrix multiplication:$ 12.123 $s, for threads %d , for matrix size %d x %d
MKL (%2.3f) + CUBLAS (%2.3f) - Total Time for matrix multiplication:$ 1.354 $s, for threads %d , for matrix size %d x %d"""

for size in range(1000,9000,1000):
    for threads in range (0,17,4): # cannot alter threads value for next iteration, it stores its own copy
        if threads == 0:
            threads = 1
        scale = 0.05
        j=0
        tmpres_list = []
        # tmpres = ""

        print size,threads,1
        # print (i,(k-j), "  ",k,j)
        # ws.write(size,threads,scale)
        ws.write(i,0,size) # row-dynamic, col1-threads, col2-scale, col3-exec_time
        ws.write(i,1,threads)

        tmpres = ""

        result = subprocess.Popen(cmd1+" "+str(threads)+" "+str(size)+" "+str(1), shell=True,cwd=os.getcwd(),stdout=subprocess.PIPE)
        result.wait()
        res1= result.stdout.read()
        res1 = res1.strip()
        f.write(res1) # python will convert \n to os.linesep
        f.write("\n")

        tmpres = res1
        tmpres = tmpres.strip()
        tmpres_list = tmpres.split("$")
        if len(tmpres_list) >1:
            # print(tmpres_list[1])
            tmpres_list[1] = float(tmpres_list[1].strip())
            ws.write(i,3,tmpres_list[1])   # scale's - execution time
            min_time = tmpres_list[1]
            if min_time < least_exec_time:
                least_exec_time = min_time
                pars_least_exec_time = [size,threads,scale]


        i=i+1




wb.save("result_info_cublas.xls")
f.close()
print "Least exec time : ",least_exec_time
print "parameters for least execution time : ",pars_least_exec_time
exit()



#usage: ./simpleCUBLAS threads size [scale] [debug]
