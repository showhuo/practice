# -*- coding: utf-8 -*
# use python3
# print
print('---input---')
name = input("What's your name? \n")
fee = 100+200
print('So you are, %s' %name)
print('Fee is, %d' %fee)   

# if else
print('---if else---')
score = 88
if score > 90:
    print('Excellent!')
else:
    if score < 60:
        print('Failed.')
    else:
        print('Not bad.')

# for loop
print('---for loop---')
sum1 = 0
for num in range(11):
    sum1 += num
print(sum1)

# while
print('---while loop---')
sum1 = 0
index = 0
while index < 11:
    sum1 += index
    index += 1
print(sum1)

# lists
print('---lists---')
lists = [1,2,3]
lists.append('a')
print('lists is: %s' %lists)
print(len(lists))
lists.insert(0,0)
lists.pop()
print(lists)

# tuples
print('---tuples---不能修改，只能取值')
tuples = (1,2,3) 
print(tuples[0])

# dictionary
print('---dictionary---')
scores = {'a':1,'b':2,'c':3}
scores['d'] = 4
print(scores)
scores.pop('a')
print('a' in scores)
print(scores.get('a',20))

# set
print('---set---')
s = set([1,2])
s.add(1)
print(s)
print('1' in s)

# function
print('---function---')
def addone(score):
    return score + 1  
print(addone(99))

# range and sum
print((range(1,100,2)))
print(sum(range(1,100,2)))

# input calculate
num = input('Input numbers, seperated by space: \n')
lists = num.split()
result = 0
for ele in lists:
    result += int(ele)
print('Sum is %d' %result)

# 正则，获取字符串首尾的数字
import re
def getNumFromLog(s):
  reg = re.compile(r'(\d+):(\w+):(\d+)')
  res = reg.match(s)
  return (int(res.group(1)),res.group(2),int(res.group(3)))