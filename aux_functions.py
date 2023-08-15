#! /usr/bin/python3

from collections import defaultdict

def merge_dicts(dicts):
    res = defaultdict(list)
    for dictionary in dicts:
        for key,values in dictionary.items():
            if isinstance(values,list):
                res[key].extend(values)
            else:
                res[key].append(values)

    return res


if __name__ == '__main__':
    #check merge_dicts
    a = {'a': [1,2,3], 'b': 2, 'c': [2,1]}
    b = {'a': 2, 'b': [12,2,2,2], 'd': 2}
    c = {'a': 2, 'b': 2, 'e': 1}
    print(merge_dicts([a,b,c]))
