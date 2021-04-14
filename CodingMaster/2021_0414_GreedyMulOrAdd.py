import sys

input = sys.stdin.readline()
input = input.rstrip('\n')
pre = 0

for i, c in enumerate(input):
    c = int(c)

    if i == 0:
        pre = c
        continue

    if pre is 0 or c is 0:
        pre = pre + c
    else:
        pre = pre * c

print(pre)