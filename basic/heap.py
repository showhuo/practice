class ListNode(object):
  def __init__(self, x=None):
    self.x = x
    self.next = None

# 合并K个有序链表
# !注意，heapq 的用法比较墨迹
import heapq
def mergeKLists(lists):
  if not lists:
    return None
  dummy = ListNode(None)
  temp = dummy
  heap = []
  # !入堆的时候，会按 tuple 顺位比较，因此需要自己维护一个递增指针，防止 val 相同的节点入堆报错
  count = 0
  for node in lists:
    if node:
      heapq.heappush(heap,(node.val,count,node))
      count += 1
  # 循环取出堆顶，取下头部节点，然后再放回堆里
  while heap:
    node = heapq.heappop(heap)[2]
    temp.next = node
    temp = temp.next
    node = node.next
    count += 1
    if node:
      heapq.heappush(heap,(node.val,count,node))
  return dummy.next

# 找出无序数组中第K大的元素
def findKthLargest(nums,k):
  # 先用小顶堆解法，维护一个 k 的小顶堆，遍历数组先将堆填满
  # 后来的元素如果比堆顶元素大，则删掉堆顶，然后入堆，即 heappushpop，最后堆顶就是答案
  PQ = []
  for i in nums:
    if len(PQ) < k:
      heapq.heappush(PQ,i)
    else:
      heapq.heappushpop(PQ,i)
  return heapq.heappop(PQ)

# 接上题，对于静态数据源，改良快排的分区思想是最快的 O(N)
import random
import sys
def findKthLargestByPartition(nums,k):
  res = None
  def quickSort(nums,left,right,k):
    nonlocal res
    if left > right:
      return
    p = partition(nums,left,right)
    if p + 1 == k:
      res = nums[p]
      return
    elif p + 1 < k:
      quickSort(nums,p+1,right,k)
    else:
      quickSort(nums,left,p-1,k)
  # 分区函数，快排的分区选点对有序数组性能不友好
  # 我们可以改为随机选点，或者打乱输入源来确保 O(N) 的复杂度
  def partition(nums,left,right):
    benchmarkIdx = random.randint(left,right)
    tar = nums[benchmarkIdx]
    i = j = left # 快慢双指针
    while j <= right:
      if nums[j] > tar:
        nums[i],nums[j] = nums[j],nums[i]
        i += 1
      j += 1
    # !注意，对于标杆不是尾巴的，我们要把标杆值当前的下标重新找出来，交换到 i 的位置
    curbenchIdx = nums.index(tar,i)
    nums[i],nums[curbenchIdx] = tar,nums[i]
    return i
  quickSort(nums,0,len(nums)-1,k)
  return res

# 给定数组 nums 和窗口大小 k，该窗口每次右移一步，求每次最大值构成的数组
def maxSlidingWindow(nums,k):
  res = []
  PQ = []
  deletedHashmap = {}
  # 默认是小顶堆，因此需要稍作改造入堆，这样堆顶取反就是最大值 -PQ[0]
  for i in range(k):
    heapq.heappush(PQ,(-nums[i],i))
  for i in range(len(nums)-k):
    # 如果堆顶已被标记删除，则需要重新堆化
    while PQ[0][1] in deletedHashmap:
      heapq.heappop(PQ)
    root = -PQ[0][0]
    res.append(root)
    val = nums[i]
    if val >= root:
      heapq.heappop(PQ)
    else:
      # !需要打上已删除的标记，因为heap直接删除指定元素成本太高
      deletedHashmap[i] = True
    heapq.heappush(PQ,(-nums[i+k],i+k))
  while PQ[0][1] in deletedHashmap:
      heapq.heappop(PQ)
  res.append(-PQ[0][0])
  return res

# 上述问题，还有进阶的 queue 解法，不大好理解
from collections import deque
def maxSlidingWindowPro(nums,k):
  res = []
  # 该队列保存潜在最大值的下标
  Q = deque()
  for i in range(len(nums)):
    # 当前窗口在 i-(k-1) 到 i 之间，下标在窗口左侧的，放弃
    if Q and Q[0] < i - (k - 1):
      Q.popleft()
    # 窗口右侧的值，循环与队尾比较，如果大于队尾，则删除尾巴
    while Q and nums[Q[-1]] < nums[i]:
      Q.pop()
    # 开始收集结果
    if i - (k - 1) >= 0:
      res.append(nums[Q.popleft()])
  return res

# 找出第 n 个 ugly number，后者的定义是只能被 2、3、5 整除
# 数字逐一检查太慢，事实上只有 2**i * 3**j * 5**k 形式的数字满足要求
# 构造一堆满足形式的数字，然后排序最快
import math
class Solution(object):
  uglyNumberCache = sorted([2**i * 3**j * 5** k for i in range(32) for j in range(32) for k in range(32)])
  def nthUglyNumber(self,n):
    return self.uglyNumberCache[n-1]

# super ugly numbers
def nthSuperUglyNumber(n,primes):
  # 参考 ugly number 的 DP 解法，每一个 un 只能由之前的 un 乘以某个 primes 数获得
  # 动态转移方程：res[next] = min([res[ti]*i,res[tj]*j,...res[tk]*k])
  # 其中 ti 是绑定素数 i 的指针，指向 ugly 数组的下标
  # 哈希表维护这些指针
  hashmap = {}
  res = [None] * n
  res[0] = 1
  for i in primes:
    hashmap[i] = 0
  # 开始 DP 递推下一个 ugly number
  for i in range(1,n):
    tempArr = [res[hashmap[k]]*k for k in primes]
    res[i] = min(tempArr)
    for k in primes:
      if res[hashmap[k]]*k == res[i]:
        # 找到上一个使用的下标，右移一位
        hashmap[k] += 1
  return res[n-1]

# !上述问题的 min 可以用小顶堆优化
def nthSuperUglyNumberPro(n,primes):
  # 使用 heap 优化的 DP 解法
  # 推导 ugly 数组的动态方程：res[1] = min(res[0]*p1,res[0]*p2...)
  # 当找到对应的 pi 时，将下标递增一位，上述的 min 用小顶堆代替即可
  res = [None] * n
  res[0] = 1
  # 用来存储 k 个下标指针
  hashmap = {}
  for k in primes:
    hashmap[k] = 0
  # 小顶堆，存储候选 ugly，这里用 tuple 存储额外的 k 信息，方便下面取栈顶时更新哈希表
  heap = [(res[hashmap[k]]*k,k) for k in primes]
  heapq.heapify(heap)
  for i in range(1,n):
    res[i] = heap[0][0]
    # 循环检查栈顶重复最小值，取出栈顶，并插入新元素
    while heap[0][0] == res[i]:
      k = heap[0][1]
      hashmap[k] += 1
      heapq.heapreplace(heap,(res[hashmap[k]]*k,k))
  return res[n-1]

  
# 动态数据源，实时计算中位数
# 经典的 heap 应用，一大一小，维持平衡，取大顶堆的 root
class MedianFinder:
    def __init__(self):
      """
      initialize your data structure here.
      """
      self.minHeap = []
      self.maxHeap = []

    def addNum(self, num: int) -> None:
      if num is None:
        return
      if not self.maxHeap and not self.minHeap:
        # 注意构造大顶堆要取反
        heapq.heappush(self.maxHeap,-num)
      else:
        if num > -self.maxHeap[0]:
          heapq.heappush(self.minHeap,num)
        else:
          # 注意构造大顶堆要取反
          heapq.heappush(self.maxHeap,-num)
        # 重新检查两个堆的比例
        self.balanceTwoHeap()

    def balanceTwoHeap(self):
      while len(self.maxHeap) - len(self.minHeap) > 1:
        node = -heapq.heappop(self.maxHeap)
        heapq.heappush(self.minHeap,node)
      # 总数为奇数得很，保持大顶堆个数多一个
      while len(self.minHeap) - len(self.maxHeap) >= 1:
        node = heapq.heappop(self.minHeap)
        heapq.heappush(self.maxHeap,-node)

    def findMedian(self) -> float:
      leng = len(self.minHeap) + len(self.maxHeap)
      if leng % 2 != 0:
        return -self.maxHeap[0]
      else:
        return (-self.maxHeap[0] + self.minHeap[0]) / 2

# 接上题，如果数据源总是在 0-100 之间，怎么优化？
# 可以考虑桶排序或计数排序，但未必比 heap 快


def topKFrequent(nums,k):
  # 经典 Top K 问题
  # 静态数据源可以用快排分区
  hashmap = {}
  cache = []
  res = None
  # 遍历一遍，统计各元素次数
  for i in nums:
    if i not in hashmap:
      hashmap[i] = 0
    hashmap[i] += 1
  # 遍历哈希表，构造新数组（因为直接遍历哈希表无法保证按照 i 顺序遍历）
  for i in hashmap:
    cache.append((i,hashmap[i]))
  # 快排
  def quickSort(cache,left,right):
    nonlocal res
    if left > right:
      return
    p = partition(cache,left,right)
    if p + 1 == k:
      res = [i[0] for i in cache[:k+1]]
    elif p + 1 < k:
      quickSort(cache,p+1,right)
    else:
      quickSort(cache,left,p-1)
  # 分区函数，随机参照物
  def partition(cache,left,right):
    slow = fast = left
    val = cache[right][1]
    while fast < right:
      if cache[fast][1] > val:
        cache[slow], cache[fast] = cache[fast], cache[slow]
        slow += 1
      fast += 1
    cache[right],cache[slow] = cache[slow],cache[right]
    return slow
  # 执行
  quickSort(cache,0,len(cache)-1)
  return res

import heapq
class Twitter:
  # 核心是 heapq.merge / merge k sorted lists 思想
  # 维护一个 size len(followee) 的 heap，不断取堆顶元素构造成目标数组
  def __init__(self):
      """
      Initialize your data structure here.
      """
      # 哈希表存储用户 id 与其关注对象
      self.follows = {}
      # 第二个哈希表，存储用户发的 tweet
      self.newsFeed = {}
      self.time = 0 # 全局时间指针，入堆优先级

  def postTweet(self, userId: int, tweetId: int) -> None:
      """
      Compose a new tweet.
      """
      if userId not in self.newsFeed:
        self.newsFeed[userId] = []
      self.newsFeed[userId].append((self.time,userId,tweetId))
      self.time -= 1

  def getNewsFeed(self, userId: int):
      """
      Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
      """
      # 初始化关注者，防止空指针报错
      if userId not in self.follows:
        self.follows[userId] = set([userId])
      #! 从关注者中分别取出最近发布的一条 tweet 构造 heap
      # 注意我们启用第四个指针，倒序追踪 tweet 数组下标
      h = []
      for p in self.follows[userId]:
        if p in self.newsFeed and self.newsFeed[p]:
          time,uid,twit = self.newsFeed[p][-1]
          h.append((time,uid,twit,len(self.newsFeed[uid])-1))
      heapq.heapify(h)
      res = []
      # 循环操作堆顶， heap-pop-push 即 heapreplace
      count = 0
      while h and count < 10:
        time,uid,tweetId,idx = h[0]
        res.append(tweetId)
        count += 1
        # 对应博主的 tweet 数组还有存货，取最近一条，入堆
        if idx > 0:
          newTime,uid,newTw = self.newsFeed[uid][idx-1]
          heapq.heapreplace(h,(newTime,uid,newTw,idx-1))
        else:
          heapq.heappop(h)
      return res

  def follow(self, followerId: int, followeeId: int) -> None:
      """
      Follower follows a followee. If the operation is invalid, it should be a no-op.
      """
      if followerId not in self.follows:
        self.follows[followerId] = set([followerId])
      self.follows[followerId].add(followeeId)

  def unfollow(self, followerId: int, followeeId: int) -> None:
      """
      Follower unfollows a followee. If the operation is invalid, it should be a no-op.
      """
      if followerId not in self.follows:
        self.follows[followerId] = set([followerId])
      if followeeId in  self.follows[followerId]  and followeeId != followerId:
        self.follows[followerId].discard(followeeId)


# 给定两个有序数组，各取一个元素组成对，取 k 对 (u,v) 使得总和最小
# DP 和 Top K 思路都可以解
# 最优解法是：转化为 merge k sorted list 问题
# 维护一个 size len1 或 len2 的小顶堆，不断取堆顶元素构造目标数组
def kSmallestPairs(nums1,nums2,k):
  h = []
  res = []
  # 构造一个 size len1 的小顶堆
  count = 0
  for i in nums1:
    # 指针3和指针4为了追踪具体的 list
    h.append((i+nums2[0],count,i,0))
    count += 1
  heapq.heapify(h)
  while len(res) < k and h:
    jdx = h[0][3]
    i,j = h[0][2],nums2[jdx]
    res.append([i,j])
    if jdx < len(nums2) - 1:
      heapq.heapreplace(h,(i+nums2[jdx+1],count,i,jdx+1))
      count += 1
    else:
      heapq.heappop(h)
  return res

# IPO 初始资金 W，每个项目所需资金和回报数组，只能做 k 个项目，求最大收益
def findMaximizedCapital(k,W,Profits,Capital):
  # 用回报数组构造一个大顶堆，成员是 (-pVal,cVal)，回报相同的选成本最小的。
  # Greedy 思想，每次检查本金是否足以开发最大收益的项目，即取出堆顶
  # 暂不满足条件的堆顶，构成另一个大顶堆
  # 做完当前项目后，本金增加，从新的大顶堆开始寻找项目
  # 重复上述过程
  h = []
  for i in range(len(Profits)):
    h.append((-Profits[i],Capital[i]))
  heapq.heapify(h)
  count = 0
  while True:
    newH = []
    # 本金不够
    while h and h[0][1] > W:
      newH.append(heapq.heappop(h))
    # 本金够了，做最大收益的项目
    if h and count < k:
      W += -h[0][0]
      count += 1
      heapq.heappop(h)
      # 将没做的项目继续合并
      for i in newH:
        heapq.heappush(h,i)
    else:
      return W

# 给定字符串数组，求有序的 Top K frequent，次数相同的要求优先小字符
# 看着像 Top K，其实不是，因为所求的结果数组是有序排列的，需要正常构造大顶堆取值
import collections
def topKFrequentStr(words,k):
  # Counter API 快速统计个数，产出哈希表
  count = collections.Counter(words)
  h = [(-freq,w) for w, freq in count.items()]
  heapq.heapify(h)
  return [heapq.heappop(h)[1] for _ in range(k)]

# 给定数组nums，两个元素可成对，求绝对差值第 K 小
def smallestDistancePair(nums,k):
  h = []
  leng = len(nums)
  for i in range(leng-1):
    for j in nums[i+1:]:
      if len(h) < k:
        # 大顶堆
        heapq.heappush(h,-abs(nums[i]-j))
      else:
        # 神烦大顶堆比较
        if abs(nums[i]-j) < -h[0]:
          heapq.heapreplace(h,-abs(nums[i]-j))
  return -h[0]

# 上述解法虽然正确，但是大量的遍历和比较
# 事实上不需要比较全部，可以先将数组排序，然后转化为 merge k sorted list 问题
# 维护一个 size len(nums)-1 的小顶堆
# 元素 tuple 需要维护当前 list 以及 list 当前比较元素的信息
def smallestDistancePairPro(nums,k):
  nums = sorted(nums)
  leng = len(nums)
  h = []
  count = 0
  for i in range(leng-1):
    # i 表示当前 list，1 表示当前位置与 list 头部的下标距离
    h.append((abs(nums[i]-nums[i+1]),i,1))
  heapq.heapify(h)
  while count < k-1:
    _,i,indent = h[0]
    # 当前 list 还有元素可以使用
    if i+indent < leng-1:
      heapq.heapreplace(h,(abs(nums[i]-nums[i+indent+1],i,indent+1)))
    else:
      heapq.heappop(h)
    count += 1
  return h[0][0]

# 上述解法虽然不错，但最坏情况空间复杂度仍然不理想（堆太大）
# 将数组排序后，可以用变种的二分查找，大幅度尝试
# 差值一定落在区间 [0,max-min]，每次取 mid 值，然后从头遍历一遍
# 差值小于 mid 的数量，如果小于 k，差值区间取右半边，重复上述过程
#! 复杂的 kth 问题，该解法具有一定的普适性，一来不需要计算所有可能，二来不需要维护额外的堆
def smallestDistancePairProMax(nums,k):
  nums = sorted(nums)
  n = len(nums)
  l = 0
  r = nums[-1] - nums[0]
  while l <= r:
    count = 0
    mid = l + (r - l)//2
    # 从新遍历计数，差值小于 mid 的有几对
    j = 0
    for i in range(n):
      while j < n and nums[j] - nums[i] <= mid:
        j += 1
      count += j - i - 1
    if count < k:
      l = mid + 1
    else:
      r = mid - 1
  return l