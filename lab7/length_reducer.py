"""Подсчитывает количество слов каждой длины."""
import sys
from itertools import groupby
from operator import itemgetter


def tokenize_input():
    """Разбивает каждую строку стандартного ввода на ключ и значение. """
    for line in sys.stdin:
        yield line.strip().split('\t')

    # Построить пары "ключ-значение" из длины слова и счетчика, разделенные табуляцией


for word_length, group in groupby(tokenize_input(), itemgetter(0)):
    try:
        total = sum(int(count) for word_length, count in group)
        print(word_length + '\t' + str(total))
    except ValueError:
        pass  # Если счетчик не является целым числом, то слово игнорируется
