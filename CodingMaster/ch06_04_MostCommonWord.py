import re
import collections

def solution(paragraph, banned):
    paragraph = paragraph.replace(',','')
    paragraph = paragraph.replace('.','')

    paragraph = paragraph.lower()
    splitted_text = paragraph.split(' ')

    word_dict={}

    for w in splitted_text:
        if w in banned:
            continue
        elif w in word_dict:
            word_dict[w] = word_dict[w]+1
        else:
            word_dict[w] = 1

    res = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)

    return res[0]

def book_solution(paragraph, banned):
    words = [word for word in re.sub(r'[^\w]', ' ', paragraph).lower().split()
             if word not in banned]

    #Counter는 해시 가능한 객체를 세기 위한 dict 서브 클래스입니다.
    # 요소가 딕셔너리 키로 저장되고 개수가 딕셔너리값으로 저장되는 컬렉션입니다.
    # 개수는 0이나 음수를 포함하는 임의의 정숫값이 될 수 있습니다.
    # Counter 클래스는 다른 언어의 백(bag)이나 멀티 셋(multiset)과 유사합니다.
    counts= collections.Counter(words)

    # index 의미 [1] 최빈값중에 No 1. most_common함수는 최빈값의 자료형을 tuple형태로 반환하는데
    # key value가 오므로 key값을 추출해야햐 하므로 [0][0]을 넣었다.
    return counts.most_common(1)[0][0]


print(solution('Bob hit a ball, the hit BALL flew far after it was hit.', ['hit']))