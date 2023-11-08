from utils.function import add

# 文字列に含まれているアルファベットの数を数える関数
def count_word(s, word):
    assert isinstance(s, str)
    assert isinstance(word, str) and len(word) == 1
    count = 0
    for c in s:
        if c == word:
            count = add(count, 1)
    return count
