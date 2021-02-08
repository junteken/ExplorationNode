import sys

def stack_push(inp):
    stack_data.append(inp)

def stack_pop():
    if stack_empty():
        return -1
    else:
        return stack_data.pop()

def stack_empty():
    return 1 if len(stack_data) == 0 else 0

def stack_top():
    if stack_empty():
        return -1
    else:
        return stack_data[-1]

def stack_size():
    return len(stack_data)

cmd_dict={'push': stack_push, 'top': stack_top, 'size': stack_size,
          'empty': stack_empty, 'pop': stack_pop}
stack_data=[]

N = int(sys.stdin.readline())

for i in range(N):
    cmd_str = sys.stdin.readline()

    if len(cmd_str.split()) > 1: # push명령어
        cmd_dict[cmd_str.split()[0]](cmd_str.split()[1])
    else:
        print(cmd_dict[cmd_str.split()[0]]())





