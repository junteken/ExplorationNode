import sys

def queue_push(inp):
    stack_data.append(inp)

def queue_pop():
    if queue_empty():
        return -1
    else:
        pop = stack_data[0]
        del stack_data[0]
        return pop

def queue_empty():
    return 1 if len(stack_data) == 0 else 0

def queue_top():
    if queue_empty():
        return -1
    else:
        return stack_data[-1]

def queue_size():
    return len(stack_data)

def queue_front():
    if queue_empty():
        return -1
    else:
        return stack_data[0]

def queue_back():
    if queue_empty():
        return -1
    else:
        return stack_data[-1]

cmd_dict={'push': queue_push, 'pop': queue_pop, 'size': queue_size,
          'empty': queue_empty, 'front': queue_front, 'back': queue_back}
stack_data=[]

N = int(sys.stdin.readline())

for i in range(N):
    cmd_str = sys.stdin.readline()

    if len(cmd_str.split()) > 1:# push명령어
        cmd_dict[cmd_str.split()[0]](cmd_str.split()[1])
    else:
        print(cmd_dict[cmd_str.split()[0]]())




