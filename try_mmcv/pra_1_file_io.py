import mmcv

# # load data from a file
# data = mmcv.load('test.json')
# data = mmcv.load('test.yaml')
# data = mmcv.load('test.pkl')
# # load data from a file-like object
# with open('test.json', 'r') as f:
#     data = mmcv.load(f)
#
# # dump data to a string
# json_str = mmcv.dump(data, file_format='json')
#
# # dump data to a file with a filename (infer format from file extension)
# mmcv.dump(data, 'out.pkl')
#
# # dump data to a file with a file-like object
# with open('test.yaml', 'w') as f:
#     data = mmcv.dump(data, f, file_format='yaml')

a1 = mmcv.list_from_file("asset/a.txt")
a2 = mmcv.list_from_file("asset/a.txt", offset=2)
a3 = mmcv.list_from_file("asset/a.txt", max_num=2)
a4 = mmcv.list_from_file("asset/a.txt", prefix='/mnt/')
print(a1)
print(a2)
print(a3)
print(a4)

b1 = mmcv.dict_from_file("asset/b.txt")
b2 = mmcv.dict_from_file("asset/b.txt", key_type=int)
print(b1)
print(b2)
