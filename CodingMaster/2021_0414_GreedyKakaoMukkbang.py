import sys

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

result = foodfood(food_times, k)

print(result)






