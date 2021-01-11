
import collections

def solution(str_list):
    ordered_str = [''.join(sorted(w)) for w in str_list]
    print(ordered_str)
    group = collections.defaultdict(list)
    group = {}
    for i, char_list in enumerate(ordered_str):
        if char_list in group:
            temp = group[char_list]
            temp.append(str_list[i]) # append함수는 return이 None이여서 이렇게 분리해서코딩
            group[char_list] = temp
            # group[char_list] = group[char_list].append(str_list[i])
        else:
            group[char_list] = [str_list[i]]

    print(group)



anagram=['eat', 'tea', 'tan', 'ate', 'nat', 'bat']

# print(solution(anagram))
solution(anagram)
