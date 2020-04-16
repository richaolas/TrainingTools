import json
# totoal = []
# a = {}
# a['b'] = [[1,2,3,4],[1,2,3,5]]
# print(a)
# f = open('result.txt','w')
#
# totoal = [a,a,a,a,a,]
#
# f.write('[\n')
# for idx, item in enumerate(totoal):
#     if idx > 0:
#
#         f.write(',\n' + json.dumps(item))
#     else:
#         f.write(json.dumps(item))
# f.write(']')
#
# f.close()

f = open('coco_train_2017.txt.json', 'r')
x = json.loads(f.read())
for i in range(10):
    print(x[i])

#print(x)
#
# for k in x:
#     print(x[k])

