def solution(arr):
    answer = []
    # [실행] 버튼을 누르면 출력 값을 볼 수 있습니다.
    temp = None
    for i in arr:
        if i is not temp:
            answer.append(i)
            temp = i
        else:
            continue

    return answer


p1 = [1, 1, 3, 3, 0, 1, 1]
p2 = [4, 4, 4, 3, 3]

print(solution(p1))
print(solution(p2))