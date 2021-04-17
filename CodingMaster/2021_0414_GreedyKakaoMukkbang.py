import sys
import heapq

foo = sys.stdin.readline()
foo = foo.lstrip('[')

foos, k = foo.split(']')
k = k.lstrip()
k = int(k)
food_times = [int(t) for t in foos.split(',')]

def foodfood(food_times, k):

    tick = 0

    while k > 0:
        if sum(food_times) == 0:
            return -1

        if food_times[tick%len(food_times)] > 0:
            food_times[tick%len(food_times)] -= 1
            tick += 1
            k -= 1
        else:
            # foodtime이 0이면 먹을게 없어서 다음으로 넘긴다.
            tick += 1

    if sum(food_times) == 0:
        return -1

    while food_times[tick%len(food_times)] == 0:
        tick += 1

    return tick%len(food_times)+1

def optimalfood(food_times, k):
    if sum(food_times) <= k:
        return -1

    q=[]
    for i in range(len(food_times)):
        heapq.heappush(q, (food_times[i], i+1))

    sum_value = 0 # 먹기위해 사용한 시간
    previous = 0 # 직전에 다 먹은 음식 시간

    length = len(food_times) # 남은 음식의 개수

    # sum_value + (현재의 음식 시간 - 이전 음식 시간)* 현재 음식 개수와 k 비교

    while sum_value + ((q[0][0] - previous)*len)



result = foodfood(food_times, k)

print(result)






