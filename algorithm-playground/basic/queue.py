# 数组实现队列
class MyQueue(object):
  def __init__(self, size):
    self.size = size
    self.items = [None] * size
    self.head = None
    self.tail = None
  def enqueue(self, val):
    if self.tail == self.size:
      # tail 已经到头，尝试往前搬运一次
      if self.head > 0:
        for i in range(self.tail - self.head):
          self.items[i] = self.items[i+self.head]
        self.tail = self.tail - self.head
        self.head = 0
      else:
        raise ValueError('The queue is full')
    self.items[self.tail] = val
    self.tail += 1
  def dequeue(self):
    if self.head == self.tail:
      raise ValueError('The queue is empty')
    val = self.items[self.head]
    self.head += 1
    return val

# 链表实现队列
class Node(object):
  def __init__(self, val=None):
    self.val = val
    self.next = None

class MyQueueLL(object):
  def __init__(self):
    self.tail = self.head = Node()
  def enqueue(self, val):
    node = Node(val)
    self.tail.next = node
    self.tail = node
    return self.head
  def dequeue(self):
    if not self.head.next:
      raise ValueError('The queue is empty')
    val = self.head.next.val
    self.head = self.head.next
    return val

# 数组实现循环队列
class LoopQueue(object):
  def __init__(self, size):
    self.items = [None] * size
    self.size = size
    # 标记当前队列的真实大小
    self.count = 0
    self.head = self.tail = 0
  def enqueue(self, val):
    pass
  def dequeue(self):
    pass

# Task Scheduler 任务调度，要求至少每隔 n 个时间才能执行相同任务，这期间可以执行别的任务，或者发呆
# 求执行完所有任务所需的最少时间
# 有两种可能，一是 n 比较小，不需要发呆，任务可以串联执行，此时答案是 len(tasks)
# 需要发呆的情况，将各字符出现的次数进行统计，排序取得最大次数 K，然后分成 K 份，它们之间距离都是 n
# 将字符按次数从大到小，依次填充到 k 份小集合里，直到没有字符，此时它们的长度是 (n+1)*(k-1) + 最后一份大小
def leastInterval(tasks,n):
  arr = [0]*26 # 最多记录26个字符出现的次数
  for c in tasks:
    arr[ord(c) - ord('A')] += 1
  arr.sort() # 数组尾巴就是最大次数，有可能重复
  k = arr[25]
  i = 25
  while i > 0 and arr[i] == k:
    # 尝试找出最大次数相同的情况，这决定了最后一份大小，有多少个最大次数的元素，就是最后一份的大小，不需要 idle
    # 这么倒序计算 i 是因为效率高
    i -= 1
  lastKsize = 25 - i
  return max([len(tasks), (n+1) * (k-1) + lastKsize])

  # 上题跟队列关系不大，不过也可以用优先级队列解决

# Shortest Subarray with Sum > K 最短连续子序列，和大于 K
# 这里我们用到 monoqueue 单调队列
from collections import deque
def shortestSubarray(A,K):
  leng = len(A)
  if A[0] >= K:
    return 1
  res = leng + 1
  sumArr = [0]*leng # 用求和数组更方便求解，因为携带了求和的信息
  sumArr[0] = A[0]
  for i in range(1,leng):
    sumArr[i] = sumArr[i-1] + A[i]
  monoque = deque() # 单调递增队列存储下标，遍历 sumArr 且每次都与队列的首尾元素比较
  for j in range(leng):
    # 假如 tail 比较大，即使存在后续下标 z 使得 z - tail > K，那么 z - j 肯定大于 K，显然 j 的位置更近，tail 可以丢弃
    while monoque and sumArr[monoque[-1]] >= sumArr[j]:
      monoque.pop()
    # 假设 j - head >= K，因为单调递增，head 之后的序列相加肯定更大，head 可以丢弃，此时要记录一次可能的 res
    while monoque and sumArr[j] - sumArr[monoque[0]] >= K:
      res = min([res,j - monoque.popleft()])
    monoque.append(j)
  return res if res != leng + 1 else -1

# 最近3秒内的调用次数
class RecentCounter(object):
  def __init__(self):
    self.queue = deque()
  def ping(self, t):
    # 将时间减去3秒，循环与队列 head 比较
    while self.queue and self.queue[0] < t - 3000:
      self.queue.popleft()
    self.queue.append(t)
    return len(self.queue)