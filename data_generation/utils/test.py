# print(len(VIDEO_CLIPS['train']))
# print(len(VIDEO_CLIPS['val']))
# print(len(VIDEO_CLIPS['test']))

# def compare(v1, v2):
#     b1 = int(v1[5:6])
#     b2 = int(v2[5:6])
#     if b1 == b2:
#         l1 = int(v1[7:8])
#         l2 = int(v2[7:8])
#         if l1 == l2:
#             count1 = int(v1[2:4])
#             count2 = int(v2[2:4])
#             if count1 == count2:
#                 if 'In' in v1:
#                     return -1
#                 return 1
#             return count1 - count2
#         return l1 - l2
#     return b1 - b2
#
# l = VIDEO_CLIPS['train'] + VIDEO_CLIPS['test'] + VIDEO_CLIPS['val']
# from functools import cmp_to_key
# l.sort(key=cmp_to_key(compare))
#
# for f in l:
#     print(f)