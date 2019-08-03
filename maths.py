import math
# 用给定的硬币组成目标金额，只能用递归解法


def combineCoins(sum, coins):
    def calculate(resultArr, left):
        if left == 0:
            print(resultArr)
            return
        elif left < 0:
            return
        for value in coins:
            left = left - value
            resultArr1 = resultArr.copy() # 必须复制一份，尝试添加
            resultArr1.append(value)
            calculate(resultArr1, left)
    calculate([], sum)

# combineCoins(10,[1,2,3,4])

# 求目标数字整除的所有可能


def divideCombines(num):
    def calculate(resultArr, left):
        if left == 1:
            if 1 not in resultArr:
                resultArr.append(1)
            print(resultArr)
            return
        if left < 1:
            return
        for idx in range(1, num+1):
            if idx == 1 and 1 in resultArr:
                continue
            newArr = resultArr.copy()
            if left % idx == 0:
                newArr.append(idx)
                calculate(newArr, left/idx)
    calculate([], num)


# divideCombines(8)

# 分治思想下的归并排序


def mergeSort(to_sort=[]):
    if len(to_sort) <= 1:
        return to_sort
    mid = math.floor((len(to_sort)/2))
    left = to_sort[0:mid]
    right = to_sort[mid:]
    left = mergeSort(left)
    right = mergeSort(right)
    return merge(left, right)


def merge(listA, listB):
    result = []
    while len(listA) > 0 and len(listB) > 0:
        if listA[0] < listB[0]:
            result.append(listA[0])
            listA.pop(0)
        else:
            result.append(listB[0])
            listB.pop(0)
    if len(listA) > 0:
        result += listA
    if len(listB) > 0:
        result += listB
    return result


print(mergeSort([5, 1, 4, 3, 2]))
