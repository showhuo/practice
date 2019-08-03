import math
#! 归并排序，分治思想
# 辅助函数，合并两个有序数组
# a - 有序数组，b - 有序数组
def merge(a,b):
    arr = []
    while len(a) > 0 and len(b) > 0:
        if a[0] <= b[0]:
            arr.append(a[0])
            a.pop(0)
        else :
            arr.append(b[0])
            b.pop(0)
    # 剩余的有序数组放在尾巴即可
    if len(a) > 0:
        arr += a
    if len(b) > 0:
        arr += b
    return arr
# 主函数
# source - 待排序的数组
def mergeSort(source):
    # 终点
    if len(source) == 1:
        return source
    length = len(source)
    # n=k-1 成立，求 n=k
    mid = math.floor(length/2)
    leftList = source[0:mid]
    rightList = source[mid:]
    left = mergeSort(leftList)
    right = mergeSort(rightList)
    result = merge(left,right)
    return result
# 测试
# sortedArr = mergeSort([1,3,2,5,4,7,6])
# print(sortedArr)

#! 排列，从数组中按一定顺序取 4 位密码，可重复取值
arr = ['a','b','c','d','e']
# 递归
def getPwd(size,str):
    if size == 0:
        print(str)
        return str
    else:
        # 假设 n=k 成立，求 n=k-1，这与假设 n=k-1 成立求 n=k 逻辑类似，本质都是逼近。
        for ele in arr:
            newStr = str + ele
            getPwd(size-1,newStr)
# getPwd(4,'')
# 循环
def getPwdByLoop():
    for i in arr:
        for j in arr:
            for k in arr:
                for l in arr:
                    print(i+j+k+l)
# getPwdByLoop()

#! 排列，按一定顺序从 n 个元素中取 m 个，不重复取。可以穷举出所有可能，在概率中应用广泛。
# source - 剩余数组，result - 当前排列，m - 目标数量，其中 len(source) >= m
count = 0
def permutation(source, result, m):
    # 取值完成
    if len(result) == m:
        print(result)
        global count
        count += 1
        return
    # n=k 时成立，求 n=k-1
    for ele in source:
        newResult = result.copy()
        newResult.append(ele)
        # 不重复取，剔除刚取到的 ele
        newSource = source.copy()
        newSource.remove(ele)
        # 继续排列
        permutation(newSource,newResult,m)
# permutation(arr, [], 5)
# print('count is %d' %count)

#! 组合，不在乎顺序从 n 个元素中取 m 个，不重复取。在词组分析、多维度数据分析中应用广泛。
# source - 剩余数组，result - 当前排列，m - 目标数量，其中 len(source) >= m
combineArr = []
def combination(source,result,m):
    # 组合完成
    if len(result) == m:
        global count
        count += 1
        combineArr.append(result)
        return 
    # n=k 成立，求 n=k-1
    for idx in range(len(source)):
        newResult = result.copy()
        newResult.append(source[idx])
        # 更严格的参数范围缩减
        newSource = source.copy()
        del newSource[0:idx + 1]
        combination(newSource,newResult,m)
# combination(arr,[],3)
# print('combineArr is %s' %combineArr)

# TODO 10人抽奖，依次抽取三等奖3人，二等奖2人，一等奖1人，不重复中奖
# source - 剩余待抽数组，result - 中奖数组
def lottery(source,result):
    # 终止条件
    if len(result) == 6:
        print(result)
        return
    # n=k 成立，求 n=k-1
    for idx in range(len(source)):
        newResult = result.copy()
        newResult.append(source[idx])
        # 缩减参数范围
        newSource = source.copy()
        del newSource[0:idx+1]
        lottery(newSource,newResult)
# arr = list(range(10))
# lottery(arr,[])

# 二分法变形
# arr - 有重复元素的有序数组，求最后一个值为 value 的元素位置
# 循环
def bsearchByLoop(arr,value):
    if value < arr[0] or value > arr[-1]:
        return -1
    n = len(arr)
    low = 0
    high = n - 1
    while low <= high:
        mid = low + math.floor((high - low)/2)
        if arr[mid] < value:
            low = mid + 1
        elif arr[mid] > value :
            high = mid - 1
        else:
            if mid == n - 1 or arr[mid+1] != value:
                return mid
            else:
                low = mid + 1
    return -1
# print(bsearchByLoop([1,2,3,4,4,4],4))
# 递归
# 其中 low 与 high 是渐变的
def bsearchByRecur(source,value,low,high):
    if low > high:
        return -1
    mid = low + math.floor((high-low)/2)
    midValue = source[mid]
    if midValue < value:
        low = mid + 1
        return bsearchByRecur(source,value,low,high)
    elif midValue > value:
        high = mid - 1
        return bsearchByRecur(source,value,low,high)
    else:
        if mid == len(source) - 1 or source[mid+1] != value:
            return mid
        else:
            low = mid + 1
            return bsearchByRecur(source,value,low,high)
# print(bsearchByRecur([1,3,5,6,7,7,7],7,0,6))

# 查找第一个大于等于 value 的元素位置
def bsearchBigThanValue(arr,value):
    n = len(arr)
    low = 0
    high = n - 1
    while low <= high:
        mid = low + math.floor((high-low)/2)
        if arr[mid] < value:
            low = mid + 1
        elif arr[mid] > value:
            high = mid - 1
        else:
            if mid == 0 or arr[mid-1] < value:
                return mid
            else:
                high = mid - 1
    return -1
print(bsearchBigThanValue([1,3,4,5,5,5],5))


