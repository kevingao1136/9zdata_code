# #%%
# from typing_extensions import IntVar


# def romanToInt(s: str) -> int:

#     d = {
#         'I':1,
#         'V':5,
#         'X':10,
#         'L':50,
#         'C':100,
#         'D':500,
#         'M':1000
#     }

#     left = 0
#     right = 1
#     res = 0

#     while right <= len(s) - 1:
#         if d[s[right]] <= d[s[left]]:
#             res += d[s[left]]
#             left += 1
#             right += 1

#         else:
#             res += (d[s[right]] - d[s[left]])
#             left += 2
#             right += 2

#     if left == len(s) -1:
#         res += d[s[left]]

#     return res

# romanToInt('LVIII')
# # %%
# max(['a','b','c','d','ee'])
# %%
def isValid(s: str) -> bool:

    d = {
        '(':')',
        ')':'(',
        '{':'}',
        '}':'{',
        '[':']',
        ']':'['
    }

    seen = []

    for i, cur in enumerate(s):
        if d[cur] not in seen:
            seen.append(d[cur])
        else:
            seen.pop(seen.index(d[cur]))

    return seen

isValid('()')
# %%
