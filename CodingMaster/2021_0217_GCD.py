import sys

def GCD(a, b):
    if a> b:
        big= a
        small = b
    else:
        big= b
        small = a

    while(True):
        remainder = big%small
        if remainder == 0:
            return small

        big = small
        small = remainder

def GCD_Sum(numlist):
    sum = 0

    for i in range(0, len(numlist) - 1):
        for j in range(i+1, len(numlist)):
            sum=sum+GCD(numlist[i], numlist[j])

    return sum


# print(GCD_Sum([10, 20, 30, 40]))

N = int(sys.stdin.readline())
for i in range(N):
     cmd_str = sys.stdin.readline()
     int_str= cmd_str.split()
     num_list=[]

     for i in range(0, int(int_str[0])):
         num_list.append(int(int_str[i+1]))

     print(GCD_Sum(num_list))




