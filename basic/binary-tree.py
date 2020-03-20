class TreeNode(object):
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None

# preOrder 前序遍历：中-左-右
def preOrder(tree):
  if not tree:
    return None
  print(tree)
  preOrder(tree.left)
  preOrder(tree.right)

# inOrder 中序遍历：左-中-右
def inOrder(tree):
  if not tree:
    return 
  inOrder(tree.left)
  print(tree)
  inOrder(tree.right)

# postOrder 后序遍历：左-右-中
def postOrder(tree):
  if not tree:
    return 
  postOrder(tree.left)
  postOrder(tree.right)
  print(tree)

# traverseByLevel 按层遍历，需要借助队列
from collections import deque
def traverseByLevel(tree):
  q = deque([tree])
  while q:
    node = q.popleft()
    print(node.val)
    if node.left:
      q.append(node.left)
    if node.right:
      q.append(node.right)

# 按层遍历就是 BFS，它有一些经典的变种问题：
# 每一层独立构成数组
# 每一层第一个元素/最后一个元素/最大值/最小值

# binary search tree 二叉查找树，有序
# 查找
def bstFind(tree,val):
  if not tree:
    return None
  if tree.val == val:
    return tree
  if tree.val < val:
    tree = tree.right
  else:
    tree = tree.left
  return bstFind(tree,val)

# 插入
def bstInsert(tree,val):
  node = tree
  while node:
    if node.val < val: # 对于支持重复元素的BST，可以放到右侧，查找时稍作修改，继续往右查找
      if node.right:
        node = node.right
      else:
        node.right = TreeNode(val)
        return 
    elif node.val > val:
      if node.left:
        node = node.left
      else:
        node.left = TreeNode(val)
        return

# 删除，需要区分子节点数量，有两个子节点的话需要将右子节点的最小节点找出来，替换到被删除节点的位置
# 也可以虚拟删除，标记额外属性，查找和插入时稍作修改
def bstDelete(tree,val):
  parent = None
  node = tree
  while node and node.val != val:
    if node.val < val:
      parent = node
      node = node.right
    else:
      parent = node
      node = node.left
  if not node:
    return None
  # 有两个子节点，需要找出右边的最小子节点
  if node.left and node.right:
    miniParent = node
    mini = node.right
    while mini.left:
      miniParent = mini
      mini = mini.left
    # 将 mini 节点放到 node 位置，这里我们只替换 val 值，简化代码
    node.val = mini.val
    # !重定向目标指针，删除 mini 节点
    node = mini
    parent = miniParent
  # 没有子节点或只有一个子节点
  child = None
  if node.left:
    child = node.left
  elif node.right:
    child = node.right
  if not parent: # 删除的是根节点
    tree = child
  if parent.left == node:
    parent.left = child
  elif parent.right == node:
    parent.right = child

# 找前驱节点，即小于 val 的最大的节点，本质是中序遍历的中间过程
# 需要区分与父节点的左右关系
def bstFindFront(tree,val):  
  pass

# 比较两棵树
def isSameTree(p,q):
  if p is None and q is None:
    return True
  elif p is None:
    return False
  elif q is None:
    return False
  return isSameTree(p.left,q.left) and isSameTree(p.right,q.right) and p.val == q.val

from collections import deque
# 上述问题的循环解法，按层遍历的思路
def isSameTreeLoop(p,q):
  d = deque([(p,q)])
  # 辅助函数，判断两个节点值相等
  def checkEq(p,q):
    if not p and not q:
      return True
    if not p or not q:
      return False
    if p.val != q.val:
      return False
    return True
  while d:
    node1,node2 = d.popleft()
    if not checkEq(node1,node2):
      return False
    if node1:
      d.extend([(node1.left,node2.left),(node1.right,node2.right)])
  return True

# 判断一棵树是否对称
# 与比较两棵树类似，递归反向比较左右
def isSymmetric(root):
  if not root:
    return True
  # 辅助递归函数
  def recur(left,right):
    if not left and not right:
      return True
    if not left or not right:
      return False
    if left.val != right.val:
      return False
    return recur(left.left,right.right) and recur(left.right,right.left)
  return recur(root.left,root.right)

# 尝试非递归解法，关键点：d.append((p.left,q.right))
def isSymmetricByLoop(root):
  if not root:
    return True
  d = deque([(root,root)])
  while d:
    p,q = d.popleft()
    if not p and not q:
      return True
    if not p or not q:
      return False
    if p.val != q.val:
      return False
    if p:
      d.append((p.left,q.right))
      d.append((p.right,q.left))
  return True

# 计算树的高度
def maxDepth(root):
  # 辅助递归函数
  def recur(root,h):
    if not root:
      return h - 1
    return max([recur(root.left,h+1),recur(root.right,h+1)])
  return recur(root,1)

# 从下往上，按层遍历
# 维护两个指针，分别对应 prev 数组和 cur 数组
# 每次遍历 prev 产生 cur 数组
# 然后 res.appendleft(prev)，prev = cur，直到循环结束
def levelOrderBottom(root):
  prev = [root]
  res = deque()
  while prev:
    cur = []
    for node in prev:
      if node.left:
        cur.append(node.left)
      if node.right:
        cur.append(node.right)
    res.appendleft(map(lambda x: x.val,prev))
    prev = cur
  return res

# !将有序数组转换为尽可能平衡 BST
# 二分查找的思想，因为 BST 中序遍历是有序的
# 数组的 mid 值应该作为 root 节点，左半边的 mid 应该作为左侧 root，右侧同理
def sortedArrayToBST(nums):
  # 辅助递归函数
  def recur(nums,left,right):
    if left > right:
      return
    mid = left + ((right - left) >> 1)
    node = TreeNode(nums[mid])
    node.left = recur(nums,left,mid-1)
    node.right = recur(nums,mid+1,right)
    return node
  return recur(nums,0,len(nums)-1)

# 判断一棵树是否平衡，高度差不超过 1
def isBalanced(root):
  res = True
  # 辅助函数，类似求🌲高度
  def calHeight(root,h):
    nonlocal res
    if not root:
      return h-1
    left = calHeight(root.left,h+1)
    right = calHeight(root.right,h+1)
    if abs(left-right) > 1:
      res = False
    return max([left,right])
  calHeight(root,1)
  return res

# 求🌲的最小高度
# 与求最大高度类似
def minDepth(root):
  if not root:
    return 0
  left = minDepth(root.left)
  right = minDepth(root.right)
  # tricky part
  if left == 0 or right == 0:
    return left+right+1
  return min([left,right]) + 1

# 给定一棵树和数值sum，判断能否找到路径，使得加和等于sum
def hasPathSum(root,sum):
  res = False
  # 辅助递归函数，不返回值
  # 找到外部指针 res 想要的结果时，停止递归
  def traceBack(root,sum):
    nonlocal res
    if not root:
      return
    sum -= root.val
    if sum == 0:
      res = True
      return
    traceBack(root.left,sum)
    traceBack(root.right,sum)
  traceBack(root,sum)
  return res

# 翻转二叉树，homebrew 作者与 Google 的梗
def invertTree(root):
  if not root:
    return
  root.left, root.right = invertTree(root.right), invertTree(root.left)
  return root

# 给定普通 binary tree，寻找两个节点的最近公共祖先 LCA
def lowestCommonAncestor(root,p,q):
  res = None
  # !辅助递归函数，一次判断 p 和 q
  def containOne(root,p,q):
    nonlocal res
    if not root:
      return False
    mid = p.val == root.val or q.val == root.val
    left = containOne(root.left,p,q)
    right = containOne(root.right,p,q)
    # tricky part：两个节点不在同一边
    if left + right + mid == 2:
      res = root
    return left or mid or right
  containOne(root,p,q)  
  return res

# 给定 bst，寻找两个节点的最近公共祖先 LCA
# 与常规二叉树相比，bst 有大小分区，路径可以更明确
def bstLCA(root,p,q):
  if not root:
    return
  if root and root.val > p.val and root.val > q.val:
    # 都比当前节点小，往左边找
    return bstLCA(root.left,p,q)
  elif root and root.val < p.val and root.val < q.val:
    return bstLCA(root.right,p,q)
  else:
    return root

# 给定有序数组[1,2...,n]，求所有可能的 BST
def generateTrees(n):
  # 递归函数，产出 left 到 right 能组成的全部 BST 数组
  def createBSTs(left,right):
    res = []
    if left > right:
      res.append(None)
      return res
    # 取 i 作为 root 节点，相当于分解为 right-left 个子问题
    for i in range(left,right+1):
      leftList = createBSTs(left,i-1)
      rightList = createBSTs(i+1,right)
      for leftNode in leftList:
        for rightNode in rightList:
          node = TreeNode(i)
          # 构造当前 BST
          node.left = leftNode 
          node.right = rightNode
          res.append(node)
    return res
  return createBSTs(1,n)

# 接上题，如果不需要求具体组合，只求数量
# 参考上述思路可推导 DP 转移方程：g(n) = g(0)*g(n-1) + g(1)*g(n-2) +...+ g(n-1)*g(0)
# 其中 g(0) == g(1) == 1
# 这里使用哈希表加递归实现便于理解，而非 DP
hashmap = {}
def numTrees(n):
  if n == 0 or n == 1:
    return 1
  if n in hashmap:
    return hashmap[n]
  res = 0
  for i in range(n):
    res += numTrees(i)*numTrees(n-i-1)
  hashmap[n] = res
  return res

import sys
# 校验 BST
# 尝试中序遍历过程中判断
def isValidBST(root):
  res = True
  prev = -sys.maxsize
  # 辅助递归函数，中序遍历与检查
  def recur(root):
    nonlocal res,prev
    if not root:
      return 
    recur(root.left)
    if prev < root.val:
      prev = root.val
    else:
      res = False
      return
    recur(root.right)
  recur(root)
  return res

# 按层遍历变种，要求产出每一层的数组，上面我们已经求解过反序的了。。。
# 进一步可要求奇偶数反序
def levelOrder(root):
  res = []
  queue = deque([root])
  odd = True
  while queue:
    res.append(list(map(lambda x: x.val,queue if odd else reversed(queue))))
    odd = not odd
    temp = deque()
    while queue:
      node = queue.popleft()
      if node.left:
        temp.append(node.left)
      if node.right:
        temp.append(node.right)
    queue = temp
  return res

# 给定前序和中序遍历所得的数组，构造唯一的目标 BT
# 根据前序遍历的定义，preorder 头部元素一定是 root 节点
def buildTree(preorder,inorder):
  # 注意这里的终止条件必须是 inorder 数组
  if not inorder:
    return 
  root = TreeNode(preorder.pop(0))
  idx = inorder.index(root.val)
  # 从 inorder 数组找出下标，左侧就是左树，右侧就是右树
  root.left = buildTree(preorder,inorder[:idx])
  root.right = buildTree(preorder,inorder[idx+1:])
  return root

# 与上题类似，给定后序和中序遍历数组，构造 BT
# postorder 尾巴元素是 root 节点
def buildTree2(inorder,postorder):
  hashmap = {}
  for i in range(len(inorder)):
    hashmap[inorder[i]] = i
  def recur(inorder,postorder):
    nonlocal hashmap
    if not inorder or not postorder:
      return
    root = TreeNode(postorder.pop())
    idx = hashmap[root.val]
    # 注意，这里必须先构造右树
    root.right = recur(inorder[idx+1:],postorder)
    root.left = recur(inorder[:idx],postorder)
    return root
  return recur(inorder,postorder)

# 给定一颗 BT 和整数 sum，求所有能够求和为 sum 的路径（要求终点是最后一层）
def pathSum(root,n):
  res = []
  # 辅助函数
  def findPath(root,n,tempArr):
    if not root:
      return
    n -= root.val
    tempArr.append(root.val)
    if n == 0 and not root.left and not root.right:
      res.append(tempArr)
    findPath(root.left,n,tempArr[:])
    findPath(root.right,n,tempArr[:])
  findPath(root,n,[])
  return res

# 拍平一棵树，要求节点顺序是前序遍历的顺序
# 进阶：要求使用 right 指针连接
def flatten(root):
  # 先尝试用额外的 stack 实现
  node = root
  stack = []
  while node:
    if node.right:
      stack.append(node.right)
      node.right = None
    if not node.left:
      node.left = stack.pop()
    node = node.left
  # 现在我们获得了 left 指针连起来的链表
  # 然后我们把它改成 right 指针。。。
  node = root
  while node:
    node.right = node.left
    node.left = None
    node = node.right
  return root

# 上述问题，再进阶：使用 right 指针连接，并且要求不使用额外空间
# TODO 该方法比较烧脑
# 使用 right -> left -> root 非常规遍历顺序，可以看作前序遍历的相反版本
def flattenPro(root):
  # prev 指针，储存递归函数的返回值
  prev = None
  # 递归函数，返回要求的树节点
  def recur(root):
    nonlocal prev
    if not root:
      return
    # 按照顺序，将右树、左树按照要求，用 right 指针排好序
    recur(root.right)
    recur(root.left)
    # 将 right 指针指向排序好的子树
    root.right = prev
    root.left = None
    prev = root
    return root
  return recur(root)

# 给定一颗完全二叉树，要求每个节点添加 next 指针指向正右方
def connect(root):
  # BFS + queue 可解
  pass

# 接上题，不使用额外空间的解法
def connectPro(root):
  # 递归解法
  if not root:
    return
  if root.left:
    root.left.next = root.right
    if root.next:
      root.right.next = root.next.left
  connectPro(root.left)
  connectPro(root.right)

# 如果给的不是完全二叉树，只是普通树
def connectNormal(root):
  # !先用 BFS + queue + 哨兵解法
  d = deque([root,None])
  while d:
    node = d.popleft()
    if node: # 使用 None 间隔开每一行
      node.next = d[0]
      if node.left:
        d.append(node.left)
      if node.right:
        d.append(node.right)
    elif d: # tricky part  
      d.append(None)
  return root

# 找出所有路径，求它们之和
def sumNumbers(root):
  res = []
  # 辅助递归函数
  def recur(root,temp):
    nonlocal res
    if not root:
      return
    temp += str(root.val)
    if not root.left and not root.right:
      res.append(temp)
    recur(root.left,temp)
    recur(root.right,temp)
  recur(root,'')
  return sum(map(lambda x: int(x),res))

# queue + stack + loop 实现 postorder 遍历
def postorderTraversal(root):
  res = deque()
  stack = [root]
  while stack:
    node = stack.pop()
    # !tricky part
    res.appendleft(node.val)
    if node.left:
      stack.append(node.left)
    if node.right:
      stack.append(node.right)
  return res

# 给定一棵树，从右边观察，列出从上到下每一层看到的第一个元素
def rightSideView(root):
  # 首先想到的是按层遍历的变种问题
  Q = deque([root,None])
  res = []
  while Q:
    node = Q.popleft()
    if node:
      # 利用哨兵快速定位每一行的尾巴
      if Q and Q[0] is None:
        res.append(node.val)
      if node.left:
        Q.append(node.left)
      if node.right:
        Q.append(node.right)
    elif Q: # 添加最后的哨兵
      Q.append(None)
  return res

# 给定一颗完全二叉树，快速计算它的节点数量
# 普通的遍历需要 O(N)，完全二叉树可以做到 logN 平方，需要先比较左右高度
def countNodes(root):
  if not root:
    return 0
  # 比较左右高度，相等则说明左侧是满二叉树，反之右侧是满二叉树
  # 满二叉树的节点数，可以通过 2**n - 1 快速获得
  leftH = calH(root.left)
  rightH = calH(root.right)
  if leftH == rightH:
    return pow(2,leftH) - 1 + countNodes(root.right)
  else:
    return pow(2,rightH) - 1 + countNodes(root.left)

# 辅助函数，计算高度
def calH(root):
  if not root:
    return 0
  return 1 + calH(root.left)

# 跳节点求和，要求不允许直连的节点
# 不能等同于奇数偶数层求和
# DP 抉择
def rob(root):
  # 辅助递归函数
  def recur(root,hashmap):
    if not root:
      return 0
    if root in hashmap:
      return hashmap[root]
    # 分别算出1、3层之和，以及2层之和
    res13 = root.val
    res2 = recur(root.left,hashmap) + recur(root.right,hashmap)
    if root.left:
      res13 += recur(root.left.left,hashmap) + recur(root.left.right,hashmap)
    if root.right:
      res13 += recur(root.right.left,hashmap) + recur(root.right.right,hashmap)
    hashmap[root] = max([res13,res2])
    return hashmap[root]
  return recur(root,{})

# 找出所有最左侧的节点之和
def sumOfLeftLeaves(root):
  res = 0
  # 辅助函数
  def recur(root):
    nonlocal res
    if not root:
      return
    if root.left and not root.left.left and not root.left.right:
      res += root.left.val
    recur(root.left)
    recur(root.right)
  recur(root)
  return res

# N 叉树，按层遍历
def levelOrderN(root):
  res = [[]]
  Q = deque([root,None])
  while Q:
    node = Q.popleft()
    if node:
      res[-1].append(node.val)
      if node.children:
        Q.extend(node.children)
    else:
      # 遇到换行哨兵，新增一个尾巴
      res.append([])
      if Q:
        Q.append(None)
  return res

# 不一定从 root 开始，但要求连续，找出所有和为 n 的路径
# !需要两个递归
def pathSumP(root,n):
  if not root:
    return 0
  count = countNormal(root,n)
  countLeft = pathSumP(root.left,n)
  countRight = pathSumP(root.right,n)
  return count + countLeft + countRight

# 辅助函数
def countNormal(root,n):
  if not root:
    return 0
  minuN = n - root.val
  return (1 if minuN == 0 else 0) + countNormal(root.left,minuN) + countNormal(root.right,minuN)

# 序列化和反序列化
def serialize(root):
  pass

# 每个子树的所有节点求和，求出现次数最多的和
def findFrequentTreeSum(root):
    res = []
    hashmap = {}
    # 辅助函数，计算树所有节点的和，用哈希表缓存
    def calSum(root):
      nonlocal hashmap
      if not root:
        return 0
      if root in hashmap:
        return hashmap[root]
      hashmap[root] = calSum(root.left) + calSum(root.right) + root.val
      return hashmap[root]
    # 随便什么序遍历，构造 hashmap
    def inorder(root):
      if not root:
        return
      inorder(root.left)
      calSum(root)
      inorder(root.right)
    inorder(root)
    # 改造成存储 sum: 次数 的哈希表
    newHashMap = {}
    for k in hashmap:
      tempSum = hashmap[k]
      if tempSum not in newHashMap:
        newHashMap[tempSum] = 0
      newHashMap[tempSum] += 1
    # 从 newHashMap 找到最大值对应的key
    # maxKey = max(hashmap,key=hashmap.get)
    maxV = max(newHashMap.values())
    for k in newHashMap:
      if newHashMap[k] == maxV:
        res.append(k)
    return res
    
# 倾斜度
def findTilt(root):
  # 优化两次递归为一次递归
  if not root:
    return 0
  res = 0
  # 求和的过程中，同时操作 res
  def calSum(root):
    nonlocal res
    if not root:
      return 0
    left = calSum(root.left)
    right = calSum(root.right)
    res += abs(left-right)
    return left + right + root.val
  calSum(root)
  return res

# 判断 t 是否为 s 的 subtree
def isSubtree(s,t):
  # 在 s 中找到与 t 值相同的节点
  tempArr = []
  def preOrder(root):
    nonlocal tempArr
    if not root:
      return
    if root.val == t.val:
      tempArr.append(root)
    preOrder(root.left)
    preOrder(root.right)
  preOrder(s)
  # 同时遍历两棵树比较
  def cmp(s,t):
    if not s and not t:
      return True
    if not s or not t:
      return False
    if s.val != t.val:
      return False
    return cmp(s.left,t.left) and cmp(s.right,t.right)
  # 将数组中所有节点都检查一遍
  for node in tempArr:
    if cmp(node,t):
      return True
  return False

# N叉树的前序遍历，stack解法，从右到左入栈，然后取栈顶元素继续
def preOrderN(root):
  res = []
  stack = [root]
  while stack:
    node = stack.pop()
    res.append(node.val)
    if node.children:
      for c in reversed(node.children):
        if c:
          stack.append(c)
  return res

# N 叉树的后序遍历，stack + queue 解法
def postOrderN(root):
  res = deque()
  stack = [root]
  while stack:
    node = stack.pop()
    # 骚包的左侧入队列
    res.appendleft(node.val)
    if node.children:
      for c in node.children:
        if c:
          stack.append(c)
  return res

# 括号辅助 bt 的序列化
# TODO 反序列化怎么做
def tree2str(t):
  if not t:
    return ''
  if not t.left and not t.right:
    return str(t.val)
  if not t.right:
    return str(t.val) + '(' + tree2str(t.left) + ')'
  return str(t.val) + '(' + tree2str(t.left) + ')' + '(' + tree2str(t.right) + ')'

# 合并两棵树
def mergeTrees(t1,t2):
  if not t1:
    return t2
  if not t2:
    return t1
  t1.val += t2.val
  t1.left = mergeTrees(t1.left,t2.left)
  t1.right = mergeTrees(t1.right,t2.right)
  return t1

