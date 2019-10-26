import sys
# binary search 二分查找算法,针对的是有序数组
# 四个经典的变种问题，有序数组中可能存在重复元素
# 查找第一个值等于给定值的元素
def bsFindIndex(arr,val):
  if arr[0] == val:
    return 0
  low = 0
  high = len(arr) - 1
  while low <= high:
    mid = low + (high-low)//2
    if arr[mid] == val:
      # 找到其中一个值，检查它的左边，如果相等，继续二分
      if arr[mid-1] != val:
        return mid
      else:
        high = mid - 1
    elif arr[mid] > val:
      high = mid - 1
    else:
      low = mid + 1
  return None

# 查找最后一个值等于给定值的元素，与上述类似
def bsFindLastIndex(arr,val):
  low = 0
  high = len(arr) - 1
  if arr[high] == val:
    return high
  while low <= high:
    mid = low + (high-low)//2
    if arr[mid] == val:
      # 找到其中一个值，检查它的右边
      if arr[mid+1] != val:
        return mid
      else:
        low = mid + 1
    elif arr[mid] > val:
      high = mid - 1
    else:
      low = mid + 1
  return None

# 查找第一个 >= 给定值的元素
def bsFindGreaterIndex(arr,val):
  if arr[0] >= val:
    return 0
  low = 0
  high = len(arr) - 1
  while low <= high:
    mid = low + (high - low) // 2
    if arr[mid] < val:
      low = mid + 1
    else: # 大于等于 val 的同时检查左边
      if arr[mid-1] < val:
        return mid
      else:
        high = mid - 1
  return None

# 查找最后一个小于等于给定值的元素
def bsFindLastSmallerIndex(arr,val):
  leng = len(arr)
  if arr[-1] <= val:
    return leng
  low = 0
  high = leng
  while low <= high:
    mid = low + ((high - low) >> 1)
    if arr[mid] > val:
      high = mid - 1
    else:
      # 检查右边
      if arr[mid+1] > val:
        return mid
      else:
        low = mid + 1
  return None

# 找出目标值在数组的下标，或者插入数组后应得的下标
def searchInsert(nums,target):
  if nums[0] >= target:
    return 0
  low = 0
  high = len(nums) - 1
  while low <= high:
    mid = low + ((high-low)>>1)
    val = nums[mid]
    if val == target:
      return mid
    elif val > target:
      high = mid - 1
    else:
      if mid == len(nums)-1 or nums[mid+1] > target:
        return mid + 1
      else:
        low = mid + 1

# 平方根的整数位
import math
def mySqrt(x:int) -> int:
  if x == 0:
    return 0
  low = 1
  high = x
  while low <= high:
    mid = low +((high-low)>>1)
    if mid*mid > x:
      high = mid - 1
    else:
      if (mid+1)*(mid+1) > x:
        return math.floor(mid)
      low = mid + 1

# 有序数组中寻找和为给定值的两个元素
def twoSum(numbers,target):
  leng = len(numbers)
  hashmap = set(numbers)
  for i in range(leng):
    temp = target - numbers[i]
    if temp in hashmap:
      idx2 = bsFindLastIndex(numbers,temp)
      return [i+1,idx2+1]
  return None

# 给一个API判断版本好坏，用最少的调用次数判断第一个损坏的版本
def isBadVersion(v):
  pass
def firstBadVersion(n):
  low = 1
  high = n
  while low <= high:
    mid = ((low+high)>>1)
    if isBadVersion(mid):
      if mid == 1 or not isBadVersion(mid-1): # 第一个坏的，或者前一个是好的，说明当前就是所求
        return mid
      # 否则继续往前找
      high = mid - 1
    else:
      low = mid + 1
  return None

# 判断是否能完美开方
def isPerfectSquare(num):
  low = 1
  high = num
  while low <= high:
    mid = low + ((high-low)>>1) # 整除，最小间隔就是 1
    if mid*mid > num:
      high = mid - 1
    else:
      if mid*mid == num:
        return True
      else:
        low = mid + 1
  if low > high:
    return False

# 子串匹配，假设有 1 billion 个子串需要匹配
# 双指针很简洁，但只能做到 O(N)，需要换一种思路
# 预处理主串，维护一个哈希表，key 是字符，val 是字符下标组成的数组（有序）
# 子串匹配了上一个字符，那么下一个字符如果匹配，下标一定大于 prev
# 问题就转化为：有序数组中找到第一个大于等于 prev 值的元素，O(logN)
def isSubsequence(s,t):
  hashmap = {}
  for i in range(len(t)):
    cha = t[i]
    if cha not in hashmap:
      hashmap[cha] = []
    hashmap[cha].append(i)
  prev = -1 # 记录上一个匹配字符在主串对应的下标，初始设为 0 
  for j in range(len(s)):
    cha2 = s[j]
    if cha2 not in hashmap:
      return False # 哈希表中没有下标记录，说明不匹配
    arr = hashmap[cha2] 
    # 下标组成的有序数组中，查找第一个大于等于给定值的元素，需要注意的是
    # 如果相邻的字符相同，那么此次需要找到大于 prev 的值
    current = bsFindGreaterIndex(arr,prev) 
    if current is None:
      return False
    elif current == len(arr) - 1 and arr[current] == prev:
      return False
    if arr[current] == prev:
      # 注意我们要的是主串中的下标，所以这里取值  
      prev = arr[current+1]
    else:
      prev = arr[current]
  return True 

# 给定房子、加热器的坐标有序数组，求满足全覆盖的最小半径
# 对每一所房子，找出它离左右加热器的距离，取较小的那个
# 对上述计算所得的所有距离，取最大的那个
# TODO
def findRadius(houses,heaters):
  res = 0
  for i in houses:
    rightIdx = bsFindGreaterIndex(heaters,i)
    if rightIdx is not None:
      right = heaters[rightIdx] - i
      if rightIdx >= 1:
        left = i - heaters[rightIdx-1]
      else:
        left = sys.maxsize
    else:
      right = sys.maxsize
      left = i - heaters[-1]
    temp = min([left,right])
    res = max([res,temp])
  return res

# 给一个山峰型数组，找出山顶元素
def peakIndexInMountainArray(A=[]):
  low = 0
  high = len(A) - 1
  while low <= high:
    mid = low + ((high-low)>>1)
    if A[mid] < A[mid+1]:
      low = mid + 1
    else:
      if A[mid-1] < A[mid]:
        return mid
      else:
        high = mid - 1

# 有序数组被分成两段，比如 456123，求目标值下标
# 需要将问题转化为可以使用二分查找的模型
# 当 target 与 mid 在同一侧时，按正常缩减范围
# 当 target 与 mid 在不同侧时，事实上都不需要比较，只要继续往 target 那一侧缩减范围
# 但是为了简化代码的目的，我们维护一个 comparetor
def search(nums,target):
  low = 0
  high = len(nums) - 1
  # 维护一个 comparetor，目的是简化代码
  comparetor = None
  while low <= high:
    mid = low + ((high-low)>>1)
    # !如果在同一侧
    if (nums[mid] < nums[0]) == (target < nums[0]):
      comparetor = nums[mid]
    # target 在右侧，将 mid 看作 -Inf 继续模拟二分查找
    elif target < nums[0]:
      comparetor = -sys.maxsize
    else:
      comparetor = sys.maxsize
    # 开始标准的二分查找
    if comparetor == target:
      return mid
    elif comparetor < target:
      low = mid + 1
    else:
      high = mid - 1
  return -1

# 接上题，假如有序数组中有重复元素
# 当 nums[mid] 等于 nums[0] 的时候，上题的左右判断会有问题
# 需要换一种思路，每次找出有序的一半，如果target在范围内就进行二分，否则找另一半
# TODO
def searchPro(nums,target):
  pass

# 有序的二维数组（矩阵）里，检查元素是否存在
def searchMatrix(matrix,target):
  # 桶排序的思想，以及二分查找的变种问题：
  # 先找到最后一个 nums[0] 小于等于 target 的子数组，然后检查它
  low = 0
  high = len(matrix) - 1
  # 先尝试找到目标子数组
  targetArr = None
  while low <= high:
    mid = low + ((high-low)>>1)
    head = matrix[mid][0]
    if head > target:
      high = mid - 1
    else:
      if mid == len(matrix)-1 or matrix[mid+1][0] > target:
        targetArr = matrix[mid]
        break
      else:
        low = mid + 1
  if targetArr is None:
    return False
  # 正常二分查找检查是否存在
  low = 0
  high = len(targetArr) - 1
  while low <= high:
    mid = low + ((high-low)>>1)
    midVal = targetArr[mid]
    if midVal == target:
      return True
    elif midVal < target:
      low = mid + 1
    else:
      high = mid - 1
  return False

# 上题的另一种解法，（整齐的）二维矩阵下标，与一维数组下标是可以互相转换的
# 下标转换后，因此当作常规的二分查找即可
def searchMatrixPro(matrix,target):
  n = len(matrix)
  m = len(matrix[0])
  low = 0
  high = n * m - 1
  while low <= high:
    mid = low + ((high - low) >> 1)
    # 将一维数组下标转换为矩阵下标
    val = matrix[mid//m][mid%m]
    if val == target:
      return True
    elif val < target:
      low = mid + 1
    else:
      high = mid - 1
  return False

# !有序数组在某个点被切开，比如456012，求最小值
def findMin(nums):
  low = 0
  high = len(nums) - 1
  while low <= high:
    if nums[low] <= nums[high]:
      return nums[low]
    mid = low + ((high - low) >> 1)
    # 左侧有序，往右边找
    if nums[low] <= nums[mid]:
      low = mid + 1
    else:
      # mid 有可能是目标值，因此不能减1
      high = mid

# 寻找最小连续 subArray 使得 sums >= s
def findMinSubArray(nums,s):
  pass

# 给定字符串 s 和 t，寻找最小连续区间，使得区间包含 t 的所有字符
# 经典的 subArray/Sliding window 问题：双指针 + minD + hashmap
# 快指针 j 向右遍历，每当 window 满足条件时，循环尝试将慢指针 i 右移一位
# 在这过程中尝试更新 minD 以及它对应的 i 下标
# 当快指针走到最右时，流程结束，minD 及其对应的 i 即为所求
# 优化点：此题是字符串，颗粒度更细，不需要每次完整判断 window 是否满足条件，只需判断左右边界即可
# 我们维护 t 的字符哈希表以及一个计数器 counter 来实现这个优化点
def minWindow(s: str, t: str) -> str:
  minD = sys.maxsize
  hashmap = {}
  strCache = {} # t 字符哈希表
  counter = len(t) # 可以理解为不匹配度
  i = 0
  for c in t:
    if c not in strCache:
      strCache[c] = 1
    else:
      strCache[c] += 1
  # 快指针开始遍历
  for j in range(len(s)):
    if s[j] not in strCache:
      strCache[s[j]] = 0
    elif strCache[s[j]] > 0:
      counter -= 1
    # !无论当前字符是否存在 t 中，都减去 1，这样 window 中那些多余字符，值会小于 0
    strCache[s[j]] -= 1
    # 最简化判断 window 是否满足条件
    while counter == 0:
      if j - i < minD:
        minD = j - i
        hashmap[minD] = i
      # i 向右移之前，检查 s[i] 是否为匹配 t 的必须条件
      # tricky part：字符哈希表值如果小于0，说明 window 里面字符有多余，相反则不够用
      strCache[s[i]] += 1 # 上面无差别减 1，这里需要加回去，可以认为不匹配程度上升
      if strCache[s[i]] > 0:
        counter += 1 # 不匹配度 +1
      i += 1
  # 流程结束，取当前的 minD
  if minD == sys.maxsize:
    return ''
  return s[hashmap[minD]:hashmap[minD]+minD+1]

# !上述很多求最长/最短 sliding window 的变种问题，核心往往是如何高效判断 window 是否满足条件

# 二维矩阵 row、col 都是有序，检查目标值是否存在
def searchMatrix2(matrix,target):
  # 先找出最后一个 num[0] 小于等于 target 的数组 k
  low = 0
  high = len(matrix) - 1
  k = None
  while low <= high:
    mid = low + ((high - low) >> 1)
    val = matrix[mid][0]
    if val > target:
      high = mid - 1
    else:
      if mid == len(matrix) - 1 or matrix[mid+1][0] > target:
        k = mid
        break
      else:
        low = mid + 1
  if k is None:
    return False
  # 对前 k 个数组分别进行二分查找
  for i in range(k+1):
    if bsFindIndex(matrix[i],target):
      return True
  return False

# h-index 定义：有 N 篇论文被引用至少 N 次以上，则 h-index 是 N
# 定义的解法：按引用次数从大到小排序，然后遍历寻找第一个 val < index + 1 的元素（因为 val 是递减的，index 是递增的）
# 变种问题，给定从小到大排序的 citations 数组，求 h-index
# 正常二分查找 val + index == n，如果最后找不到，则 n - (high + 1) 就是答案
def hIndex(citations=[]):
  n = len(citations)
  low = 0
  high = n-1
  while low <= high:
    mid = low + ((high - low) >> 1)
    if citations[mid] + mid == n:
      return n - mid
    elif citations[mid] + mid > n:
      high = mid - 1
    else:
      low = mid + 1
  return n - (high + 1)

