# 冒泡、选择、插入都是原地排序，因此都需要进行交换或'移动'的操作
# 冒泡排序
def bubbleSort(arr=[]):
  leng = len(arr)
  for i in range(leng):
    # 每次冒泡结束，最大的值一定排好序了，因此 j 不需要走完全程
    # 进一步优化，标记可能提前结束的交换
    isSwap = False
    for j in range(leng-i-1):
      # 每次从0开始相邻比较与交换
      if arr[j+1] < arr[j]:
        arr[j+1],arr[j] = arr[j],arr[j+1]
        isSwap = True
    if not isSwap:
      break
  return arr

# 插入排序
def insertSort(arr=[]):
  length = len(arr)
  if length <= 1:
    return arr
  for i in range(1,length):
    value = arr[i]
    # !循环往前比较，如果比前方元素小，则将前方元素'后移'一位，'移动'的开销小于交换
    j = i - 1
    while j >= 0 and arr[j] > value:
      arr[j+1] = arr[j]
      j -= 1
    # 最后在目标位置替换插入 value
    arr[j+1] = value
  return arr

# 选择排序
def selectionSort(arr=[]):
  length = len(arr)
  # 将 i 换成 i+1 之后的最小值
  for i in range(length):
    minIdx = i
    # 获取最小值下标
    for j in range(i+1,length):
      if arr[j] < arr[minIdx]:
        minIdx = j
    # 交换最小值到左侧
    arr[i], arr[minIdx] = arr[minIdx], arr[i]
  return arr

# 归并排序，自下而上，先处理子问题再合并，需要额外 O(N) 空间
def mergeSort(arr,left,right):
  if right <= left:
    return [arr[left]]
  mid = (right + left)//2
  leftArr = mergeSort(arr,left,mid)
  rightArr = mergeSort(arr,mid+1,right)
  return merge(leftArr,rightArr)
# 辅助函数，合并两个有序数组
import sys
def merge(listA=[],listB=[]):
  res = []
  i = j = 0
  while i < len(listA) or j < len(listB):
    a = listA[i] if i < len(listA) else sys.maxsize # 哨兵
    b = listB[j] if j < len(listB) else sys.maxsize
    if a < b:
      res.append(a)
      i += 1
    else:
      res.append(b)
      j += 1
  return res

# 快速排序，自上而下，先分区再处理子问题，原地排序
def quickSort(arr,left,right):
  if left >= right:
    return
  p = partition(arr,left,right)
  quickSort(arr,left,p-1) # !注意分界点 p 是不允许继续参与的
  quickSort(arr,p+1,right)
  return arr
# 辅助函数，双指针交换，分区
def partition(arr,left,right):
  i = j = left # i 是慢指针，j 是快指针
  val = arr[right] # 标杆值可以随机选取，优化快排
  while j < right:
    # !小于标杆值的，交换到左边的慢指针 i
    if arr[j] < val:
      arr[i],arr[j] = arr[j],arr[i]
      i += 1
    j += 1
  # 将标杆值与 i 交换，此时下标 i 就是分界点
  arr[i],arr[right] = arr[right],arr[i]
  return i

# O(N) 时间找出数组中第 K 大元素，快排的思路
# 分界点 p 左侧已经有 p 个比它小的元素，所以它就是第 p+1 大的元素
def findKth(arr,K):
  def findKthR(arr,left,right):
    if left > right:
      return False
    p = partition(arr,left,right)
    if p == K-1:
      return p
    elif p < K-1: # 分界点太小，往右边找
      return findKthR(arr,p+1,right)
    else: # 分界点太大，往左边找
      return findKthR(arr,left,p-1)
  return findKthR(arr,0,len(arr)-1)

# 合并区间，[[1,3],[2,6],[8,10]] -> [[1,6],[8,10]]
# 先按区间start大小排序，再遍历一遍比对重叠
def mergeIntervals(intervals):
  intervals.sort(key=lambda arr: arr[0])
  length = len(intervals)
  if length == 1:
    return intervals
  res = [intervals[0]]
  for i in range(1,length):
    prev = intervals[i-1]
    arr = intervals[i]
    if arr[0] <= prev[1]:
      res.pop()
      item = [prev[0],max([arr[1],prev[1]])]
      res.append(item)
      intervals[i] = item
    else:
      res.append(arr)
  return res

# 类似上题，在一个不重叠的区间数组中，插入一个区间，求新的数组
# [[1,3],[6,9]] + [2,5] -> [[1,5],[6,9]]
def insertIntervals(intervals,newInterval):
  res = []
  length = len(intervals)
  idx = length
  for i in range(length):
    if intervals[i][0] > newInterval[0]:
      idx = i
      break
  # 将新区间插入到 idx 位置
  intervals.insert(idx,newInterval)
  # !优化过的区间合并方式
  for j in intervals:
    if not res or res[-1][1] < j[0]:
      res.append(j)
    else:
      res[-1][1] = max([res[-1][1],j[1]])
  return res

# 三种颜色排序，要求原地排序
# 因为桶/计数排序需要额外空间，我们先尝试快排
def sortColors(nums=[]):
  length = len(nums)
  if length <= 1:
      return nums
  def quickSort(nums,left,right):
    if left >= right:
      return 
    p = partition(nums,left,right)
    quickSort(nums,left,p-1)
    quickSort(nums,p+1,right)
    return nums
  def partition(nums,left,right):
    mid = (left+right)//2
    val = nums[mid]
    i = j = left
    while j <= right:
      if nums[j] < val: #! 快排的中间分界点解法，注意最后我们需要找出 val 的当前下标 idx（从 i 往后找），然后与 i 进行交换
        nums[i],nums[j] = nums[j],nums[i]
        i += 1
      j += 1
    idx = nums.index(val,i)
    nums[idx],nums[i] = nums[i],nums[idx]
    return i
  return quickSort(nums,0,length-1)

# 接上题，用计数排序 O(N)
def sortColorsByCountingSort(nums=[]):
  length = len(nums)
  if length <= 1:
      return nums
  sumList = [0] * 3 # 因为只有012三种值
  sumList[0] = nums.count(0)
  for i in range(1,3):
    sumList[i] = sumList[i-1] + nums.count(i) # !构造小于等于 nums[i] 的总个数数组
  #! tricky part 反向遍历 nums，将 res 下标 sumList[val] - 1 赋值为 nums[j]，因为小于等于 nums[j] 的个数是 sumList[val]
  res = [0] * length
  for j in range(length-1,-1,-1):
    val = nums[j]
    res[sumList[val] - 1] = val
    sumList[val] -= 1
  return res

# 接上题，不使用排序算法，通过快排的分区思想解决
def sortColorsPro(nums=[]):
  i = j = 0
  length = len(nums)
  while j <= length:
    if nums[j] < 1:
      nums[i],nums[j] = nums[j],nums[i]
      i += 1
    j += 1
  j = i
  while j <= length:
    if nums[j] < 2:
      nums[i],nums[j] = nums[j],nums[i]
      i += 1
    j += 1
  
# !three-way-partitioning 三向分区
def sortColorsPlus(nums,mid):
  i = 0 # 左指针
  j = 0 # 主指针
  n = len(nums)-1 # 右指针
  while j <= n:
    if nums[j] < mid: # 往左扔
      nums[i],nums[j] = nums[j],nums[i]
      i += 1
      j += 1
    elif nums[j] > mid: # 往右扔，但因为不知道新的 j 大小关系，所以不能进行 j+1 而是留到下一轮
      nums[j],nums[n] = nums[n],nums[j]
      n -= 1
    else: # 等于中间值的情况
      j += 1


# 插入排序链表版
class Node(object):
  def __init__(self, x=None):
    self.val = x
    self.next = None
# 由于链表找 next 比找 prev 简单，我们将插入排序的思想反过来，将大的值往右循环比较和移动？
# 不好模拟，建议转为数组进行
def insertionSortList(head):
  pass

# 归并排序链表版
def mergeSortLL(head):
  if not head or not head.next:
    return head
  mid = findMid(head)
  rightSide = mid.next
  mid.next = None
  left = mergeSortLL(head)
  right = mergeSortLL(rightSide)
  return mergeLL(left,right)

# 辅助函数，找链表中间节点
def findMid(head):
  slow = head
  fast = head
  while fast.next and fast.next.next:
    slow = slow.next
    fast = fast.next.next
  return slow
# 辅助函数，合并两个有序链表
def mergeLL(left,right):
  dummy = cur = Node()
  while left or right:
    leftVal = left.val if left else sys.maxsize
    rightVal = right.val if right else sys.maxsize
    if leftVal < rightVal:
      cur.next = left
      left = left.next
    else:
      cur.next = right
      right = right.next
    cur = cur.next
  return dummy.next

# Largest Number 数组元素组合成最大数字 [10,2] -> ‘210’
# ! Tricky part： x + y > y + x 可以在排序的时候让 '3' 排在 '30' 之前
class LargerNumKey(str):
    def __lt__(self,other): # 自定义比较
        return self + other > other + self

def largestNumber(nums=[]):
  res = ''
  nums = sorted(map(str,nums),key=LargerNumKey)
  for i in nums:
    res += i
  return res

# H-Index 计算 H 因子，论文被引用次数大于 h 的有至少 h 篇，取最大的 h 值
# 需要借助桶排序的思想，n+1个桶，存储的是值为桶下标的个数
def hIndex(citations=[]):
  length = len(citations)
  buckets = [0] * (length+1)
  for i in citations:
    if i <= length:
      buckets[i] += 1
    else:
      buckets[length] += 1 # 大于N的值，全部存放在最后一个桶里
  count = 0
  for j in range(length,-1,-1): # 从后往前遍历所有桶，当count >= j，j就是我们所求的h因子
    count += buckets[j]
    if count >= j:
      return j
  return 0

# Wiggle Sort 一大一小排序
def wiggleSort(nums=[]):
  nums.sort()
  half = (len(nums)-1)//2
  # !奇数偶数下标，half为分界点倒序赋值
  nums[::2],nums[1::2] = nums[half::-1],nums[:half:-1]

# 检查两数组的重复元素，只存唯一值
def intersection(nums1,nums2):
  hashmap = set(nums1)
  res = set()
  for i in nums2:
    if i in hashmap:
      res.add(i)
  return list(res)

# 存单一数组中出现的所有次数
# 先排序，然后用两个指针同时遍历比较
def intersection2(nums1,nums2):
  nums1.sort()
  nums2.sort()
  i = 0
  j = 0
  res = []
  while i < len(nums1) and j < len(nums2):
    if nums1[i] == nums2[j]:
      res.append(nums1[i])
      i += 1
      j += 1
    elif nums1[i] > nums2[j]:
      j += 1
    else:
      i += 1
  return res

# 匹配字典中最长单词，由已知字符串 S 删减字母而来
# 不需要排序，核心是辅助函数
def findLongestWord(s,d=[]):
  curStr = ''
  for k in d:
    if cmpStr(s,k):
      if len(k) > len(curStr) or (len(k) == len(curStr) and k < curStr):
        curStr = k
  return curStr

# 辅助函数，比较字符串 b 是否可由 a 删减而来
def cmpStr(a,b):
  i = 0
  j = 0
  while i < len(a) and j < len(b):
    if a[i] == b[j]:
      i += 1
      j += 1
    else:
      i += 1
  # 如果 j 没走完，说明不匹配
  return j == len(b)

# 重排字符串，使其相邻字符不相同
def canReorganize(S):
  arr = [0] * 26
  for c in S:
    arr[ord(c)-ord('a')] += 1
  maxNum = max(arr)
  if maxNum > sum(arr)//2 + 1: # 如果最多的字符超过总数一半，说明无法实现
    return False
  return True

# 给出上述字符串
# 辅助函数，将统计次数的数组转化为字符从多到少排列的数组
def createStrArrByNumArr(numArr,strArr):
  if max(numArr) == 0:
    return strArr
  temp = max(numArr)
  idx = numArr.index(temp)
  c = chr(idx+97)
  while temp > 0:
    strArr.append(c)
    temp -= 1
  numArr[idx] = 0
  createStrArrByNumArr(numArr,strArr)

def reorganize(S):
  numArr = [0] * 26 # 先统计各字符出现的次数
  for c in S:
    numArr[ord(c)-ord('a')] += 1
  maxNum = max(numArr)
  if maxNum > (sum(numArr)-1)//2 + 1: # 如果最多的字符超过总数一半，说明无法实现
    return ''
  # 利用 arr 构造从多到少排序的字符数组
  strArr = []
  createStrArrByNumArr(numArr,strArr)
  # 构造 maxNum 个桶，把字符分散开
  buckets = [''] * maxNum
  for i in range(len(strArr)):
    idx = i % maxNum
    buckets[idx] += strArr[i]
  return ''.join(buckets)

# 上一题的简化解法
def reorganizePro(S=''):
  arr = sorted(S)
  arr = sorted(arr,key=arr.count,reverse=True) # 按出现次数从多到少排序
  if arr.count(arr[0]) > (len(arr)-1)//2 + 1:
    return ''
  # 打散字符，前一半字符占据偶数位
  half =(len(arr) - 1) // 2 
  arr[::2],arr[1::2] = arr[:half+1],arr[half+1:]
  return ''.join(arr)

# CarFleet 车队问题，给定两个数组分别表示汽车位置和车速，后车追上前车时会变成前车的速度，不能超车
from heapq import heappush,heappop
def carFleet(target,position,speed):
  leng = len(position)
  heap = []
  for i in range(leng):
    heappush(heap,(position[i],speed[i])) # 构造一个小顶堆
  catchCount = 0 # 追上前车的案例数量
  frontPo,frontV = heappop(heap)
  while heap:
    backPo,backV = frontPo,frontV
    frontPo,frontV = heappop(heap)
    maxTime = (target - frontPo) / frontV
    if backPo + backV * maxTime >= target: # 最大时间内能赶上前车
      catchCount += 1
  return leng - catchCount 

# 上述解法只能解决静止的问题，动态问题需要通过比对所有时间求解
def carFleetPro(target,position,speed):
  arr = sorted(zip(position,speed)) # zip 可以快速构造 tuple 迭代器，以第一个参数优先排序，有点类似迭代 heap
  times = [(target-p)/v for p,v in arr] # 所有车到达终点的时间
  res = 0
  while len(times) > 1:
    lead = times.pop()
    if lead < times[-1]: # 前车跑得快
      res += 1
    else:
      times[-1] = lead # 车速被拖慢
  return res + (1 if times else 0)

# 煎饼排序 Pancake Sort，不断翻转前 K 个元素，完成排序
def pancakeSort(A):
  res = []
  reverseMax(A,res)
  return res

# !辅助函数，将数组当前最大元素先翻转到head，再翻转到尾巴，递归执行
def reverseMax(A=[],res=[]):
  if len(A) <= 1:
    return res
  maxNum = max(A)
  idx = A.index(maxNum)
  if idx < len(A) - 1: # 最大值不在最后位置
    if idx != 0: # 不在第一位
      A[:idx+1] = reversed(A[:idx+1])
      res.append(idx+1)
    A.reverse()
    res.append(len(A))
  A.pop()
  return reverseMax(A,res)

# 找出离原点最近的前K个点
def kClosest(points,K):
  points.sort(key=lambda point: point[0]*point[0] + point[1]*point[1])
  return points[:K]

# TODO 接上题，不用排序，用快排的思想即可
def kClosestPro(points,K):
  # 分区函数，平方和小于最右数组的，都丢到左边
  def partition(arr,left,right):
    val = arr[right][0]*arr[right][0] + arr[right][1]*arr[right][1]
    i = left
    j = left
    while j < right:
      if arr[j][0]*arr[j][0] + arr[j][1]*arr[j][1] <= val:
        arr[i],arr[j] = arr[j],arr[i]
        i += 1
      j += 1
    arr[i],arr[right] = arr[right],arr[i]
    print(i)
    return i
  # 快排函数
  def quickSortPoints(arr,left,right,K):
    if left >= right:
      return 
    p = partition(arr,left,right)
    if p + 1 == K: # 找到目标K，那么数组左侧就是前K个点
      return arr[:K]
    elif p + 1 < K: # 往右边找
      quickSortPoints(arr,p+1,right,K)
    else: # 往左边找
      quickSortPoints(arr,left,p-1,K)
  # 执行
  return quickSortPoints(points,0,len(points)-1,K)

# print(kClosestPro([[1,3],[-2,2],[2,-2]],2))
# print(kClosestPro([[3,3],[5,-1],[-2,4],[1,1]],2))

# 找最大周长
def largestPerimeter(A=[]):
  A.sort()
  for i in range(len(A)-1,1,-1):
    if A[i] < A[i-1] + A[i-2]:
      return sum(A[i-2:i+1])
  return 0

# 给定一个坐标系和一个点c0，将所有点按照离c0的曼哈顿距离 |x-x0| + |y-y0| 大小排序
def allCellsDistOrder(R,C,r0,c0):
  arr = []
  for i in range(R):
    for j in range(C):
      arr.append([i,j])
  arr.sort(key=lambda point: abs(point[0]-r0) + abs(point[1]-c0))
  return arr

# 重排数组，使得相邻元素不同
def rearrangeBarcodes(barcodes=[]):
  # list.count 每次计算的话太耗时，我们先建立一个cache数组
  countArr = [0] * 10000
  for i in barcodes:
    countArr[i] += 1 # 值作为下标，计数
  barcodes = sorted(sorted(barcodes),key=lambda j: countArr[j],reverse=True) # 第一次排序是为了防止数量相同的元素乱串，第二次按数量大小排（稳定排序）
  half = (len(barcodes)-1)//2
  barcodes[::2],barcodes[1::2] = barcodes[:half+1], barcodes[half+1:]
  return barcodes
# 事实上可以不需要排序，递归取cache数组中最大值，倒推元素，然后分散到 maxNum 个单元即可

# 给定两个数组，arr2元素不重复，且arr1包含arr2，重排arr1使其元素相对位置与arr2一致，不存在的元素按大小排到队尾
import functools
def relativeSortArray(arr1=[],arr2=[]):
  hashmap = {}
  for i in range(len(arr2)):
    hashmap[arr2[i]] = i
  #!自定义比较
  def CustomCmp(a,b):
    nonlocal hashmap
    # 每次去查下标太慢了，需要维护一个额外的cache数组
    idx1 = hashmap[a] if a in hashmap else 1000 + a
    idx2 = hashmap[b] if b in hashmap else 1000 + b
    return idx1 - idx2
  arr1 = sorted(arr1,key=functools.cmp_to_key(CustomCmp))
  return arr1

# 上述自定义比较也可以用 lambda 函数实现
#  lambda a: k.get(a, 1000 + a)
