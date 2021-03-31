import numpy
# 0-1背包，n 个物品，背包最大承重 maxWeight
# 每个物品的决策：放还是不放，上一个状态集合可以推导下一个集合
# 这里有个重要的预判，即每一步骤的状态集合是有限的，不会超过总的重量限制
def getMax(items,n,maxWeight):
  # 维护一个二维数组，记录每一阶段的状态集合
  states = numpy.zeros((maxWeight,maxWeight+1),dtype=numpy.bool)
  states[0][0] = True
  if items[0] < maxWeight:
    states[0][items[0]] = True
  #! 状态转移，并且合并重复的状态
  for i in range(1,n): 
    for j in range(maxWeight): # 不放入背包
      if states[i-1][j] == True: states[i][j] = True
    for j in range(maxWeight-items[i]+1): # 放入背包
      if states[i-1][j] == True: states[i][j+items[i]] = True
  # 在最后一个阶段的状态集合里，找到目标值
  for i in range(maxWeight,0,-1):
    if states[n-1][i] == True:
      print('The max weight is %s' %i)
      return i

# getMax([2,3,4,5],4,8)

# 一维数组解法
def getMaxPro(items,n,maxWeight):
  states = numpy.zeros(maxWeight+1,dtype=numpy.bool)
  states[0] = True
  if items[0] < maxWeight: states[items[0]] = True
  for i in range(1,n):
    for j in range(maxWeight-items[i],-1,-1): # 注意一维数组解法这里只能倒序，否则会重复计算
      if states[j] == True: states[j+items[i]] = True
  for k in range(maxWeight,-1,-1):
    if states[k] == True:
      print('The max weight pro is %s' %k)
      return k

# getMaxPro([7,7,6,6,7,9],6,17)

# 求具体的组合，回溯算法示例
#! 后面会演示动态规划的倒推解法
def getCombines(items,n,maxWeight):
  theMax = getMaxPro(items,n,maxWeight)
  # i 控制递增，因为物品唯一，不能重复使用
  def recur(result,left,i):
    if left == 0: 
      print(result)
      result = []
      return
    elif left < 0:
      return 
    for j in range(i,n):
      newResult = result.copy() # 必须复制一份，尝试添加
      newResult.append(items[j]) 
      recur(newResult,left-items[j],j+1)
  print('The combination by backtracking is :')
  recur([],theMax,0)

# getCombines([1,2,3,6,7,4],6,10)

# 倒推演示求单一组合
def getCombinesByDynamic(items,n,maxW):
  states = numpy.zeros((n,maxW+1),dtype=numpy.bool)
  states[0][0] = True
  if items[0] < maxW: states[0][items[0]] = True
  for i in range(1,n):
    for j in range(maxW + 1):
      if states[i-1][j] == True: states[i][j] = True
    for j in range(maxW-items[i]+1):
      if states[i-1][j] == True: states[i][j+items[i]] = True
  target = 0
  result = numpy.zeros(5,dtype=numpy.array)
  for j in range(maxW,0,-1):
    if states[n-1][j] == True: 
      target = j
      break
  # 从目标值倒推，如果 states[i-1][target-items[i]] == True，说明物品 i 被选中了
  # 如果同时 states[i-1][target] == True，说明物品 i 可选可不选，要覆盖这种情况建议用回溯算法
  for i in range(n-1,0,-1):
    if target - items[i] > 0 and states[i-1][target-items[i]] == True:
      result.append(items[i])
      target -= items[i]
  # 上述倒推到第二行，第一行单独处理
  if target > 0: result.append(target)
  print('The combination is %s' %result)
  return result

# getCombinesByDynamic([1,2,3,6,7,4],6,10)

# 0-1背包升级版，加入物品价值
def getMaxValue(items,values,n,maxW):
  states = numpy.zeros((n,maxW+1),dtype=numpy.int8)
  states[0][0] = 0
  if items[0] < maxW: states[0][items[0]] = values[0]
  # 状态转移推导
  for i in range(1,n):
    for j in range(maxW+1):
      if states[i-1][j]: states[i][j] = states[i-1][j]
    for j in range(maxW-items[i]+1):
      # 注意，这里只能取同等重量下的最大价值，而不是被后来的组合覆盖
      if states[i-1][j] >= 0 and states[i-1][j] + values[i] > states[i][j+items[i]]: 
        states[i][j+items[i]] = states[i-1][j] + values[i]
  maxV = 0
  for i in range(maxW,0,-1):
    if states[n-1][i] > maxV: maxV = states[n-1][i]
  print('The max value is %s' %maxV)
  return maxV

# getMaxValue([1,2,3,6,7,4],[3,5,3,6,7,4],6,10)

# 扩展，购物车取价格总和刚好超过 m 的值以及组合（薅羊毛）
# 这里我们需要手动定义一个上限，比如 m + 100
def shoppingCart(values,n,targetValue):
  maxValue = targetValue + 100
  states = numpy.zeros((n,maxValue),dtype=numpy.bool)
  states[0][0] = True
  if values[0] < targetValue: states[0][values[0]] = True
  # 状态推导
  for i in range(1,n):
    for j in range(maxValue):
      if states[i-1][j] == True: states[i][j] = True
    for j in range(maxValue-values[i]):
      if states[i-1][j] == True: states[i][j+values[i]] = True
  # 找到第一个大于 targetValue 的金额
  result = 0
  for i in range(targetValue,maxValue+1):
    if states[n-1][i] == True:
      result = i
      break
  # 倒推求组合
  arr = []
  tempTotal = result
  for i in range(n-1,0,-1):
    if tempTotal-values[i] > 0 and states[i-1][tempTotal-values[i]] == True:
      arr.append(values[i])
      tempTotal -= values[i]
  if tempTotal > 0:
    arr.append(values[0])
  print('The combination is %s' %arr)
  print('The proper check bill is ￥%s' %result)
  return arr

# shoppingCart([7,7,6,6,7,9],6,17)

# 类杨辉三角，求最短路径
def getShortestRecur(i,j,mini,d,n,total):
  if i == n-1:
    if total < mini: mini = total
    print('miniDist is %s' %mini)
    return 
  if i < n and j < n:
    getShortestRecur(i+1,j,mini,d,n,total+d[i+1][j])
    getShortestRecur(i+1,j+1,mini,d,n,total+d[i+1][j+1])
  
d = [[3],[5,3],[6,2,1],[9,3,1,6]]
# getShortestRecur(0,0,100,d,4,3)

# 动态规划解法
def getShortestDynamic(d,n):
  states = [[100]*4]*4
  states[0][0] = d[0][0]
  # 状态推导，合并重复状态只取最小值
  for i in range(1,n):
    for j in range(i+1): # 选左边
      if states[i-1][j] + d[i][j] < states[i][j]:
        states[i][j] = states[i-1][j] + d[i][j]
    for j in range(i): # 选右边
      if states[i-1][j] + d[i][j+1] < states[i][j+1]:
        states[i][j+1] = states[i-1][j] + d[i][j+1]
  # 从最后一排找到目标
  mini = 100
  for i in range(n):
    if  states[n-1][i] < mini:
      mini = states[n-1][i]
  print('The miniDist is %s' %mini)
  return mini

# getShortestDynamic(d,4)

# 有1、3、5三种硬币，数量不限
# 使用最少的硬币找零钱
# 回溯，暴力搜索
least = 9
def getCoinsRecur(result,left,coins):
  global least
  if left == 0:
    if len(result) < least: least = len(result)
    return
  if left < 0:
    return
  for v in coins:
    newResult = result.copy()
    newResult.append(v)
    getCoinsRecur(newResult,left-v,coins)

# getCoinsRecur([],9,[1,3,5])
# print(least)

# 动态规划，利用状态转移方程，递归
mini = 9
cache = numpy.zeros((10,10),dtype=numpy.int8)
def getMiniCoinsDynamic(moneyLeft, count):
  global mini
  if moneyLeft == 0:
    if count < mini:
      mini = count
    return 1
  if moneyLeft < 0:
    return 0
  if cache[moneyLeft][count] > 0:
    return cache[moneyLeft][count]
  currentMini = min(getMiniCoinsDynamic(moneyLeft-1,count+1), getMiniCoinsDynamic(moneyLeft-3,count+1), getMiniCoinsDynamic(moneyLeft-5,count+1)) + 1
  cache[moneyLeft][count] = currentMini
  return currentMini

# getMiniCoinsDynamic(9,0)
# print('The miniCoins is %s' %mini)

# 借助上述结果，顺便求具体的组合

def getMiniCoins(moneyLeft, result):
  global mini
  if moneyLeft == 0:
    if len(result) == mini:
      print(result)
    return 
  if moneyLeft < 0:
    return 
  if len(result) > mini:
    return
  result1 = result.copy()
  result3 = result.copy()
  result5 = result.copy()
  result1.append(1)
  result3.append(3)
  result5.append(5)
  getMiniCoins(moneyLeft-1,result1)
  getMiniCoins(moneyLeft-3,result3)
  getMiniCoins(moneyLeft-5,result5)

# getMiniCoins(9,[])

# 编辑距离 - 莱文斯坦距离：增、删、改都会导致距离+1，距离越小越相似
# 利用 dictionary 缓存
mini = 100
a = 'mitctttmuaaa'
b = 'mttttacnuaaa'
matrix = {}
def getLwstDP(i,j,editest):
  global mini
  if i == 0 or j == 0:
    if j > 0: editest += j
    if i > 0: editest += i
    if editest < mini: 
      mini = editest
    return editest
  if 's'+str(i)+str(j)+str(editest) in matrix:
    return matrix['s'+str(i)+str(j)+str(editest)]
  if a[i] == b[j]:
    dist = min(getLwstDP(i-1,j,editest+1),getLwstDP(i,j-1,editest+1),getLwstDP(i-1,j-1,editest))
  else:
    dist = min(getLwstDP(i-1,j,editest+1),getLwstDP(i,j-1,editest+1),getLwstDP(i-1,j-1,editest+1))
  matrix['s'+str(i)+str(j)+str(editest)] = dist
  return dist

# getLwstDP(len(a)-1,len(b)-1,0)
# print('lwstDP is %s' %mini)

# 编辑距离 - 最长公共子串：增、删都会导致长度-1，越长越相似
maxLcs = 0
def getMaxLcs(i,j,longest):
  global maxLcs
  if i < 0 or j < 0:
    if longest > maxLcs:
      maxLcs = longest
    return longest
  if 's'+str(i)+str(j)+str(longest) in matrix:
    return matrix['s'+str(i)+str(j)+str(longest)]
  if a[i] == b[j]:
    current = max(getMaxLcs(i-1,j,longest),getMaxLcs(i,j-1,longest),getMaxLcs(i-1,j-1,longest+1))
  else:
    current = max(getMaxLcs(i-1,j,longest),getMaxLcs(i,j-1,longest),getMaxLcs(i-1,j-1,longest))
  matrix['s'+str(i)+str(j)+str(longest)] = current
  return current

getMaxLcs(len(a)-1,len(b)-1,0)
print('maxLcs is %s' %maxLcs)

# n个不同的数字，求最长的递增子序列长度
# 问题需要转换为：分别求出以每个元素结尾的最长子序列长度，取它们之中的最大值
arr = [3,5,1,2]
maxL = 0
cache = [0] * 9
def getMaxAscendLen(i):
  global maxL
  if i == 0:
    return 1
  if cache[i] > 0:
    return cache[i]
  result = 0
  for j in range(i-1,-1,-1):
    lastResult = getMaxAscendLen(j)
    if arr[j] < arr[i]:
      result = max(lastResult + 1, result)
    else:
      # 左边元素比较大，那么以 i 元素结尾的最长子序列就只有它自己
      result = 1
    if result > maxL:
      maxL = result
    cache[i] = result
    return result

# getMaxAscendLen(3)
# print(maxL)

def getMaxAscendLenByLoop(arr):
  length = len(arr)
  maxLen = 1
  states = [1] * length
  for i in range(1,length):
    for j in range(0,i):
      if arr[j] < arr[i]:
        states[i] = max(states[j]+1,states[i])
      else:
        continue
    if states[i] > maxLen:
      maxLen = states[i]
  print(maxLen)
  return maxLen

# getMaxAscendLenByLoop([3,5,1,2])

import math

def isPalindrome(s):
  length = len(s)
  if length == 1:
    return True
  middle = math.floor(length/2)
  i = 0
  j = length - 1
  while i < middle and j >= middle:
    if s[i] != s[j]:
      return False
    i += 1
    j -= 1
  return True

# print(isPalindrome('abba'))

# 最长回文子串
# 解法一：从两边顶点找起，DP 转移方程为 P(i,j) = True if P(i+1,j-1) == True and S[i] == S[j] else False

def longestPalindrome(s):
  states = {}
  length = len(s)
  if length <= 1:
    return s

  def isFullfilled(i,j):
    if i == j:
      return True
    if j == i+1 and s[i] == s[j]:
      return True
    if j == i+1 and s[i] != s[j]:
      return False
    if 'k' + str(i) + 's' + str(j) in states:
      return states['k' + str(i) + 's' + str(j)]
    currentStatus = isFullfilled(i+1,j-1) == True and s[i] == s[j]
    states['k' + str(i) + 's' + str(j)] = currentStatus
    return currentStatus
  
  maxLen = -1
  result = ''
  for i in range(length):
    for j in range(length-1,i-1,-1):
      if isFullfilled(i,j) == True and j - i > maxLen:
        maxLen = j - i
        result = s[i:j+1]
  return result

test = "slvafhpfjpbqbpcuwxuexavyrtymfydcnvvbvdoitsvumbsvoayefsnusoqmlvatmfzgwlhxtkhdnlmqmyjztlytoxontggyytcezredlrrimcbkyzkrdeshpyyuolsasyyvxfjyjzqksyxtlenaujqcogpqmrbwqbiaweacvkcdxyecairvvhngzdaujypapbhctaoxnjmwhqdzsvpyixyrozyaldmcyizilrmmmvnjbyhlwvpqhnnbausoyoglvogmkrkzppvexiovlxtmustooahwviluumftwnzfbxxrvijjyfybvfnwpjjgdudnyjwoxavlyiarjydlkywmgjqeelrohrqjeflmdyzkqnbqnpaewjdfmdyoazlznzthiuorocncwjrocfpzvkcmxdopisxtatzcpquxyxrdptgxlhlrnwgvee"
# print(longestPalindrome(test))

# 解法二：循环遍历，找出每个元素为中心（或伪中心）时的最长回形子串，取其中最长的，也属于 DP 范畴

# 辅助函数，检查子串长度
def check(s,i,j):
  left = i
  right = j
  while left >=0 and right < len(s) and s[left] == s[right]:
      left -= 1
      right += 1
  return right - left - 1

def longestPalindromePlus(s):
  length = len(s)
  start = 0
  end = 0
  for i in range(length):
    # 检查最长子串
    len1 = check(s,i,i)
    len2 = check(s,i,i+1)
    maxLen = max(len1,len2)
    if maxLen > end - start:
      start = i - math.floor((maxLen-1)/2)
      end = i + math.floor(maxLen/2)
  return s[start:end+1]

print(longestPalindromePlus(test))

# 给定一个数组，求连续子串的最大和
def maxSubArray0(nums):
  # 正常应该是 DP 解法，这里使用特殊的累加解法
  # 当前面之和为正数，累加到当前值，最后求数组最大值
  for i in range(1,len(nums)):
    if nums[i-1] > 0:
      nums[i] += nums[i-1]
  return max(nums)

# TODO 给定一个整数数组，求组成最大和的连续子串
# 思路：从左往右遍历一遍，当前和如果小于0则舍弃
def maxSubArray(nums):
  resultList = [nums[0]]
  currentSum = 0
  for i in nums:
    copyList = resultList.copy()
    if currentSum >= 0:
      currentSum += i
      copyList.append(i)
    else:
      currentSum = i
      copyList = [i]
    if sum(copyList) > sum(resultList):
      resultList = copyList.copy

