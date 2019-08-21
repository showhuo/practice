# 数组实现
class MyStack(object):
  def __init__(self, num):
    self.items = [None] * num
    self.size = num
    self.cur = 0
  def push(self, val):
    if self.cur >= self.size:
      raise ValueError('stack is full')
    self.items[self.cur] = val
    self.cur += 1
  # 注意，出栈只改尾巴指向，并不搬运或删除数组
  def pop(self):
    if self.cur <= 0:
      raise ValueError('stack is empty')
    val = self.items[self.cur-1]
    self.cur -= 1
    return val

# 链表实现，用头就行
class Node(object):
  def __init__(self, val=None):
    self.val = val
    self.next = None

class MyStackLL(object):
  def __init__(self):
    self.head = None
  def push(self, val):
    node = Node(val)
    node.next = self.head
    self.head = node
    return node
  def pop(self):
    if self.head is None:
      raise ValueError('stack is empty')
    val = self.head.val
    self.head = self.head.next
    return val

class Solution(object):

  # 经典的括号匹配问题
  def isValid(self, s):
    leftSet = set(['(','[','{'])
    theMap = {
      '(':')',
      '[':']',
      '{':'}'
    }
    stack = []
    for i in s:
      if i in leftSet:
        stack.append(i)
      else:
        if len(stack) == 0 or theMap[stack[-1]] != i:
          return False
        stack.pop()
    return len(stack) == 0

  # 移除最外层括号，使用经典的栈存放左括号，当栈为空时，将对应位置的括号剔除
  def removeOuterParentheses(self,S=''):
    length = len(S)
    sList = list(S)
    stack = []
    for i in range(length):
      if S[i] == '(':
        if len(stack) == 0:
          sList[i] = None
        stack.append('(')
      else:
        if len(stack) == 1:
          sList[i] = None
        stack.pop()
    sList = list(filter(None,sList))
    return ''.join(sList)

  # 进阶解法，标记多余的左括号
  def removeOuterParenthesesPro(self, S=''):
    pass

  # 构造 MinStack 使其可以 O(1) 获得 min 值
  # 每次存入的是对象 (x,min)，这样取栈顶元素就能获得 min
  def minStack(self, parameter_list):
    pass
  
# !用两个 stack 实现 queue，时间复杂度做到 O(1)
# 思路很巧妙，只有 stack2 为空，才需要搬运一次数据，“懒搬运”
class MyQueue(object):
  def __init__(self):
    self.stack1 = []
    self.stack2 = []
  # 平时只需往 stack1 存入数据
  def push(self, x):
     self.stack1.append(x)
  # 模拟从队列头部取出数据，此时尝试从 stack2 中取，如果为空，则搬运一次数据
  def pop(self):
    if not self.stack2:
      while self.stack1:
        self.stack2.append(self.stack1.pop())
    return self.stack2.pop()
  # 只取队列头部的值，不删除
  def peek(self):
    if not self.stack2:
      while self.stack1:
        self.stack2.append(self.stack1.pop())
    return self.stack2[-1]
  # 检查队列是否为空
  def empty(self):
    return len(self.stack1) == 0 and len(self.stack2) == 0

# 给定 arr 和 subArr，求 next greater value 组成的数组
def nextGreaterElement(nums1=[],nums2=[]):
  # 首先利用 stack 求出 nums2 的完整 next greater value，构造 hashmap
  stack = []
  res2 = {}
  result = []
  for i in nums2:
    while len(stack) and stack[-1] < i:
      # 栈顶的 next greate value 是 i
      res2[str(stack[-1])] = i
      stack.pop()
    stack.append(i)
  # 如果 stack 还有元素，说明没有比它们大的值了
  while len(stack):
    res2[str(stack.pop())] = -1
  # 遍历 nums1
  for j in nums1:
    result.append(res2[str(j)])
  return result

# Baseball 计分，有 int、C、D、+ 四种符号
def calPoints(arr=[]):
  stack = []
  res = 0
  for i in arr:
    if RepresentsInt(i):
      stack.append(int(i))
    elif i == 'C':
      stack.pop()
    elif i == 'D':
      stack.append(stack[-1]*2)
    elif i == '+':
      stack.append(stack[-2]+stack[-1])
  while len(stack):
    res += stack.pop()
  return res

# 辅助函数，判断是否为数字
def RepresentsInt(s):
  try: 
      int(s)
      return True
  except ValueError:
      return False

# 字符串 S、T，其中 # 表示删除，比较 S 和 T 是否相等
def backspaceCompare(S,T):
  stack1 = []
  stack2 = []
  for i in S:
    if i == '#':
      if stack1:
        stack1.pop()
    else:
      stack1.append(i)
  for j in T:
    if j == '#':
      if stack2:
        stack2.pop()
    else:
      stack2.append(j)
  if len(stack1) != len(stack2):
    return False
  while stack1 and stack2:
    if stack1.pop() != stack2.pop():
      return False
  return True

# 不断删除相邻的相同字母，像消方块那样
def removeDuplicates(S):
  stack = []
  for i in S:
    if stack and stack[-1] == i:
      stack.pop()
    else:
      stack.append(i)
  return ''.join(stack)

# 格式化 path，通过 split('/') 简化大量 if else
def simplifyPath(path):
  arr = path.split('/')
  stack = []
  setExclude = set(['.','..',''])
  for i in arr:
    if i == '..' and stack:
      stack.pop()
    elif i not in setExclude:
      stack.append(i)
  result = '/'.join(stack)
  if not result or result[0] != '/':
    result = '/' + result
  return result

# Binary Tree 的 Inorder Traversal (二叉树中序遍历)
class TreeNode(object):
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None
# 递归解法
def inorderTraversal(root):
  result = []
  def recur(root):
    if root is None:
      return 
    recur(root.left)
    result.append(root.val)
    recur(root.right)
  recur(root)
  return result
# !循环解法，非常漂亮
# 将左树压入栈，然后取栈顶，尝试将其右节点
def inorderTraversalLoop(root):
  stack = []
  result = []
  while stack or root:
    if root:
      stack.append(root)
      root = root.left
    else:
      node = stack.pop()
      result.append(node.val)
      root = node.right
  return result

# !二叉树的逐层遍历（Level Order Traversal）
# 需要一个 level 队列维护当前层级
def levelOrder(root):
  if root is None:
    return []
  result = []
  level = [root]
  while level:
    nextLevel = []
    for node in level:
      result.append(node.val)
      # 获取当前层级值，同时构造下一层
      if node.left:
        nextLevel.append(node.left)
      if node.right:
        nextLevel.append(node.right)
    level = nextLevel
  return result

# 逐层 Zigzag 来回遍历，左右左，每层单独组成小数组
def zigzagLevelOrder(root):
  if root is None:
    return []
  result = []
  level = [root]
  toRight = True
  while level:
    currentNodes = []
    nextLevel = []
    for node in level:
      if toRight:
        currentNodes.append(node.val)
      else:
        currentNodes.insert(0,node.val)
      if node.left:
        nextLevel.append(node.left)
      if node.right:
        nextLevel.append(node.right)
    result.append(currentNodes)
    toRight = not toRight
    level = nextLevel
  return result

# 二叉树的前序遍历 PreOrderTraversal
def preorderTraversal(root):
  # 递归解法
  result = []
  def recur(node):
    if node is None:
      return 
    result.append(node.val)
    recur(node.left)
    recur(node.right)
  recur(root)
  return result

# 循环栈解法，每次取栈顶的值，同时将它的右树、左树入栈
def preorderTraversalByLoop(root):
  if not root:
    return []
  result = []
  stack = [root]
  while stack:
    node = stack.pop()
    result.append(node.val)
    if node.right:
      stack.append(node.right)
    if node.left:
      stack.append(node.left)
  return result

# 反向波兰表达式法
def evalRPN(tokens=[]):
  stack = []
  for i in tokens:
    if i == '+':
      x = stack.pop()
      y = stack.pop()
      stack.append(x+y)
    elif i == '-':
      x = stack.pop()
      y = stack.pop()
      stack.append(y-x)
    elif i == '*':
      x = stack.pop()
      y = stack.pop()
      stack.append(y*x)
    elif i == '/':
      x = stack.pop()
      y = stack.pop()
      # 正负小数都转为 0
      stack.append(int(float(y)/x))
    else:
      stack.append(int(i))
  return stack[-1]

# BST Iterator，二叉搜索树的中序遍历迭代器
def BSTIterator(root):
  stack = []
  while stack or root:
    if root:
      stack.append(root)
      root = root.left
    else:
      yield stack[-1].val
      root = stack.pop().right

# 判断字符串是否为 BinaryTree 的前序遍历序列
# 如果提供树，可以考虑构造 BT preorder Iterator
# 如果不允许用树，只能用代数校验插槽数量，仅限于前序遍历
def isValidPreOrder(preorder='1,2,#,3'):
  slots = 1
  arr = preorder.split(',')
  for i in arr:
    slots -= 1
    if slots < 0:
      return False
    if i != '#':
      slots += 2
  return slots == 0

# 将嵌套的int数组拍平，flatten nested list
# 粗暴的字符串解法
def flattenByTransStr(nestedList=[]):
  theStr = str(nestedList)
  invalidSet = set(['[',']',','])
  res = []
  for i in theStr:
    if i not in invalidSet:
      res.append(i)
  filtered = list(filter(noEmpty,res))
  return list(map(int,filtered))

def noEmpty(s):
  return s != ' '

# !优雅的出入栈解法
def flattenNestedList(nestedList):
  res = []
  stack = []
  rightToStack(stack,nestedList)
  while stack:
    top = stack[-1]
    if isinstance(top,list): 
      stack.pop()
      rightToStack(stack,top)
    elif isinstance(top,int):
      res.append(top)
      stack.pop()
  return res

# 辅助函数，从右向左遍历压栈
def rightToStack(stack,L=[]):
  length = len(L)
  for i in range(length-1,-1,-1):
    stack.append(L[i])

# print(flattenNestedList([[1,2],3,[4,[5,[[[6]]]]],7]))

# 写一个简易的字符串 parser，能够解析数组
# 用 stack 暂存嵌套数组，用两个指针寻找合法字符作为数组元素
def deserialize(s=''):
  stack = []
  res = []
  i = 1
  length = len(s)
  for j in range(1,length):
    char = s[j]
    if char == '[':
      # 新建一个 list，将其同时压入栈顶数组和栈，后续引用的变化会同步
      cur = []
      stack[-1].append(cur)
      stack.append(cur)
      i = j + 1
    elif char == ',' or char == ']':
      # 检查 i:j 子串，并压入栈顶数组，前序的引用会同步发生变化
      if j > i:
        stack[-1].append(s[i:j])
      i = j + 1
      if char == ']':
        # 当前列表操作结束，已暂存至前序元素，可移除栈顶
        stack.pop()
  return res

# 解密字符串，decode string，比如：3[a2[c]] -> accaccacc
# 暂存前面的，先计算后面的，这种模型适合用 stack
# 维护一个 stack，遇到 [ 就暂存当前的字符串和数字，遇到 ] 就计算出当前字符串
def decodeString(s=''):
  stack = []
  curStr = ''
  curNum = 0
  for c in s:
    if c.isdigit():
      # 计算可能的连续数字
      curNum = 10*curNum + int(c)
    elif c == '[':
      stack.append(curNum)
      stack.append(curStr)
      curNum = 0
      curStr = ''
    elif c == ']':
      # 因为 num 与 str 一定隔着括号，这里使用之前的 num 乘以 curStr
      prevStr = stack.pop()
      num = stack.pop()
      curStr = prevStr + num*curStr
    else:
      # 普通字母
      curStr += c
  return curStr

# 从一个数字中移除 k 位，获取最小值
from collections import deque 
def removeKdigits(num='',k=0):
  # !每次入栈前，与栈顶循环比较，删除大栈顶
  if k == len(num):
    return '0'
  stack = deque()
  for i in num:
    while k and stack and int(i) < int(stack[-1]):
      stack.pop()
      k -= 1
    stack.append(i)
  if k > 0:
    while k:
      stack.pop()
      k -= 1
  while stack and stack[0] == '0':
    stack.popleft()
  return ''.join(stack) or '0'

# 判断数组中是否存在 132 pattern，位置不需要连续
# 从后往前遍历，stack 只存尽可能大的值，出栈的值赋予 third 变量，只要找到比 third 小的元素就行
def find132pattern(arr=[]):
  stack = []
  third = float('-inf')
  for i in arr.reverse():
    if i < third:
      return True
    # 栈顶元素永远是最大的，淘汰下来的给 third
    while stack and stack[-1] < i:
      third = stack.pop()
    stack.append(i)
  return False

# 循环数组寻找 next greater value
# 将 nums 延长一倍，然后用经典的 stack 方法寻找 next greater value
def nextGreaterElementsLoop(nums=[]):
  stack = []
  length = len(nums)
  res = [None]*length
  doubleNums = nums*2
  for i in range(2*length):
    while stack and doubleNums[stack[-1]] < doubleNums[i]:
      if stack[-1] < length:
        res[stack.pop()] = doubleNums[i]
      else:
        stack.pop()
    stack.append(i)
  for j in range(length):
    if res[j] is None:
      res[j] = -1
  return res

# 单线程 cpu 根据日志统计函数执行时间，可能串行、嵌套、发呆、重复执行等
# pre 记录上一条日志时间，函数已执行时间需要实时更新
def exclusiveTime(n,logs=[]):
  id,s,time = logs[0].split(':')
  id = int(id)
  time = int(time)
  stack = [id]
  prev = time
  res = [0]*n
  res[id] += time
  for i in range(1,n):
    log = logs[i]
    id,s,time = log.split(':')
    id = int(id)
    time = int(time)
    if s == 'start':
      # 更新上一个函数的执行时间，然后将当前 id 入栈，记录当前时间
      if stack:
        res[stack[-1]] += time - prev
      stack.append(id)
      prev = time
    else:
      # 更新上一个函数执行时间，出栈，记录当前时间
      res[stack.pop()] += time - prev + 1
      prev = time + 1
  return res

# 小行星碰撞
def asteroidCollision(asteroids=[]):
  stack = []
  isEq = False
  for i in asteroids:
    if i > 0:
      stack.append(i)
    else:
      while stack and stack[-1] > 0:
        # 手动控制循环
        if stack[-1] + i > 0:
          break
        elif stack[-1] + i == 0:
          stack.pop()
          isEq = True
          break
        else:
          stack.pop()
      if isEq:
        isEq = False
        continue
      if not stack  or stack[-1] < 0:
        stack.append(i)
  return stack

# 数组的 next greated value 变种问题，求下一次更高气温还有几天
def dailyTemperatures(T=[]):
  stack = []
  leng = len(T)
  res = [None]*leng
  for i in range(leng):
    val = T[i]
    while stack and T[stack[-1]] < val:
      res[stack[-1]] = i - stack[-1]
      stack.pop()
    stack.append(i)
  while stack:
    res[stack.pop()] = 0
  return res

# 计算嵌套括号的分数，() 是 1，相邻相加，嵌套则翻倍
# 与 3[a]2[c] 字符解密类似，我们需要 stack 暂存当前的计算结果
def scoreOfParentheses(s=''):
  stack = []
  cur = 0
  for i in s:
    if i == '(':
      stack.append(cur)
      cur = 0
    else:
      # !遇到 ) 则翻倍，至少是 1
      cur = stack.pop() + max([2*cur,1])
  return cur

# 另一种字符解密，a2b3 -> aabaabaab
# 算出字符串再求解的话，内存可能会爆掉
def decodeAtIdx(S='',K=1):
  # 先求总长度，通过求余数将范围缩小
  size = 0
  for i in S:
    if i.isdigit():
      size = size * int(i)
    else:
      size += 1
  # 遇到尾数数字，K 和 size 都可以缩小
  for c in reversed(S):
    if c.isdigit():
      size /= int(c)
      K %= size
    elif K == 0 or K == size:
      return c
    else:
      size -= 1
  
# Online Stock Span 今日股价比过去上涨的天数，next greater value 的高级变种
# !上一个入栈的元素，需要携带传承的信息，在它出栈的时候，将信息传递给下一个入栈元素
class StockSpanner:
  def __init__(self):
    self.stack = []
  def next(self, price):
    day = 1 # 被传递的信息
    while self.stack and self.stack[-1][0] <= price:
      day += self.stack.pop()[1]
    self.stack.append((price,day))
    return day

# 补足括号，维护两个栈，或者双指针解决
def minAddToMakeValid(S=''):
  stack1 = []
  stack2 = []
  for i in S:
    if i == '(':
      stack1.append(i)
    else:
      while stack1:
        stack1.pop()
        break
      else:
        stack2.append(i)
  return len(stack1) + len(stack2)

# 校验 pushed、popped 数组是否匹配
# 模拟 stack 的行为，先入栈，然后不断尝试出栈
def validateStackSequences(pushed=[],popped=[]):
  stack = []
  leng = len(pushed)
  i = j = 0
  while i < leng:
    c = pushed[i]
    stack.append(c)
    i += 1
    while j < leng and stack and stack[-1] == popped[j]:
      stack.pop()
      j += 1
  if stack:
    return False
  else:
    return True

# 检查字符串是否满足 abc 子串组合模式
def isValid(S='ababcc'):
  stack = []
  for i in S:
    if i == 'a' or i == 'b':
      stack.append(i)
    else:
      if len(stack) < 2 or stack[-1] != 'b' or stack[-2] != 'a':
        return False
      else:
        stack.pop()
        stack.pop()
  return len(stack) == 0

# 找最长的连续工作日，要求大于8的天数占多数
# !maxinum size subarray problem 求满足条件的最长连续子序列
# !核心是用 hashmap 记录各类分数首次出现的位置，只要目标的 score - 1 曾经出现过，那么从 score - 1 到 score 的子序列就可能是答案
# !因为这个子序列之和就是 1，满足题目所求大于 0 的最接近条件
def longestWPI(hours=[]):
  score = 0
  res = 0
  hashmap = {}
  for i in range(len(hours)):
    if hours[i] > 8:
      score += 1
    else:
      score -= 1
    # 尽可能记录当前最大长度
    if score > 0:
      res = i + 1
    # 记录首次出现该分数的位置，因为这个位置最靠前
    hashmap.setdefault(score,i)
    # 找到和是 1 的子序列，作为候选答案
    if score - 1 in hashmap:
      res = max([res, i - hashmap[score - 1]])
    return res