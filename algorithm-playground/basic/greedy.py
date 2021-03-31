# 常见算法思想之一
# 要求本次的选择，不影响后面的选择

# 给定连续N天的股票价格，每次只能买卖/持有一手，求最大收益
# 等价于找波峰和波谷，求所有最长单调递增之和
def maxProfit(prices):
  bottom = prices[0]
  res = 0
  for i in range(1,len(prices)-1):
    p = prices[i]
    if p > prices[i-1] and p >= prices[i+1]:
      res += p - bottom
    elif p <= prices[i-1] and p <= prices[i+1]:
      bottom = p
  # 最后一位
  if prices[-1] > prices[-2]:
    res += prices[-1] - bottom
  return res

# 改良版，不需要找波峰和波谷，累积连续上升的每一小步即可
def maxProfitPro(prices):
  res = 0
  for i in range(1,len(prices)):
    if prices[i] > prices[i-1]:
      res += prices[i] - prices[i-1]
  return res

# 给定字符串 S 和 T，判断 S 是否为 T 的片段，不需要连续，只要求字母相对顺序一致
# 如果是一次性查找
def isSubsequence(s,t):
  idx = -1
  for w in s:
    if t.find(w,idx+1) != -1:
      idx = t.find(w,idx+1)
    else:
      return False
  return True

# 如果是高频查找，比如有100万个S
# 先遍历一次 T，构造一个大型哈希表，记录字母对应的下标们，下标用数组存储
# 找到上一个字母对应的下标prev时，在当前字母对应的下标数组中，二分查找第一个大于prev的值
def isSubsequencePro(s,t):

  pass

# 给熊孩子分饼干，两个数组分别表示需求和供给，求能满足最多几个熊孩子
def findContentChildren(g,s):
  res = 0
  g.sort()
  s.sort()
  i = j = 0
  while i < len(g) and j < len(s):
    if s[j] >= g[i]:
      res += 1
      i += 1
      j += 1
    else:
      j += 1
  # 无论孩子剩余还是饼干剩余，都结束
  return res

# 找零钱，只有5、10、20三种，一开始没钱
# 其实可以不用哈希表，普通变量指针即可
import collections
def lemonadeChange(bills):
  hashmap = collections.defaultdict(int)
  for i in range(len(bills)):
    cur = bills[i]
    if cur == 5:
      hashmap[cur] += 1
    elif cur == 10:
      if hashmap[5] == 0:
        return False
      hashmap[5] -= 1
      hashmap[10] += 1
    else:
      if hashmap[10] == 0:
        if hashmap[5] < 3:
          return False
        hashmap[5] -= 3
      else:
        if hashmap[5] == 0:
          return False
        hashmap[10] -= 1
        hashmap[5] -= 1
  return True

# 给定一组机器人指令，左转右转前进，和一组障碍物，求最后的位置
def robotSim(commands,obstacles):
  
  pass