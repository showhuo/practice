# 单链表
class Node():
  def __init__(self,value=None,next=None):
    self.value = value
    self.next = next
  def getValue(self):
    return self.value
  def setNext(self, node):
    self.next = node
    return self
  def getNext(self):
    return self.next

nodeA = Node('a')
nodeB = Node('b')

class LinkedList(object):
  def __init__(self, head=None):
    self.head = head
    self.tail = head
  #! 性能考虑，头部添加能简洁做到 O(1)，尾部添加只能 O(n)，除非维护额外的 tail 节点
  def appendAtHead(self, value):
    newNode = Node(value)
    newNode.setNext(self.head)
    self.head = newNode
    return self
  # 尾巴添加
  def append(self, value):
    newNode = Node(value)
    self.tail.setNext(newNode)
    self.tail = newNode
    return self
  # 遍历计算大小
  def size(self):
    count = 0
    current = self.head
    while current:
      count += 1
      current = current.getNext()
    return count
  # 可视化打印
  def detail(self):
    current = self.head
    while current:
      if not current.getNext():
        print(current.getValue())
      else:
        print(current.getValue(),end='->')
      current = current.getNext()
    return self
  # 在指定节点之后插入
  def insertAfter(self, nodeRef, value):
    newNode = Node(value)
    nextNode = nodeRef.getNext()
    newNode.setNext(nextNode)
    nodeRef.setNext(newNode)
    return self
  # 在指定节点之前插入，单链表需要先找到prev节点
  def insertBefore(self, nodeRef, value):
    current = self.head
    prev = self.head
    while current and current != nodeRef:
      prev = current
      current = current.getNext()
    if current is None:
      raise ValueError('There is no target Node.')
    newNode = Node(value)
    newNode.setNext(current)
    prev.setNext(newNode)
    return self
  # 查找指定值的节点
  def search(self, value):
    current = self.head
    while current and current.getValue() != value:
      current = current.getNext()
    if current:
      return current
    return None
  # 删除指定的值
  def delete(self, value):
    prev = self.head
    current = self.head
    while current and current.getValue() != value:
      prev = current
      current = current.getNext()
    if current is None:
      raise ValueError('There is no %s to delete' %value)
    if current.getNext() is not None:
      prev.setNext(current.getNext())
    else:
      prev.setNext(None)
    return self
  # 变成循环链表
  def turnToCircle(self):
    self.tail.setNext(self.head)
    return self
  #! 反转
  def reverse(self):
    if self.size() <= 1:
      return self
    prev = None
    current = self.head
    while current is not None:
      next = current.getNext()
      current.setNext(prev)
      prev = current
      current = next  
    self.head = prev
    return self

#! 环的检测
# 将节点引用（内存地址）存入 set() 或者作为 key 存入 dict，本质都是 hashmap/hashtable
def isLoop(ll):
  current = ll.head
  theDict = {}
  while current:
    if current in theDict:
      print('There is loop in the linked list')
      return True
    theDict[current] = True
    current = current.getNext()
  print('There is no loop in the linked list')
  return False

# 合并两个有序链表：额外空间
def mergeSortedLinkedList(ll1,ll2):
  i = ll1.head
  j = ll2.head
  newNode = Node()
  newList = LinkedList()
  newList.head = newNode
  while i and j:
    if i.getValue() <= j.getValue():
      newNode.setNext(i)
      i = i.getNext()
    else:
      newNode.setNext(j)
      j = j.getNext()
    newNode = newNode.getNext()
  # 看看还剩下啥
  if i is not None:
    newNode.setNext(i)
  if j is not None:
    newNode.setNext(j)
  print(newList.detail())
  return newList

#! 合并有序链表：原地合并
def mergeSortedLinkedListPro(ll1,ll2):
  i = ll1.head
  j = ll2.head
  if i.getValue() > j.getValue():
    ll1,ll2 = ll2,ll1
  while i and j:
    while i.getNext() and i.getNext().getValue() < j.getValue():
      i = i.getNext()
    while j.getNext() and j.getNext().getValue() < i.getValue():
      j = j.getNext()
    if i.getValue() <= j.getValue():
      temp = i.getNext()
      i.setNext(j)
      i = temp
    else:
      temp = j.getNext()
      j.setNext(i)
      j = temp
  print(ll1.detail())
  return ll1

# 递归写法
# 返回剩余节点中最小的那个
def recur(i,j):
  if i is None:
    return j
  if j is None:
    return i
  if i.getValue() <= j.getValue():
    result = i
    result.setNext(recur(i.getNext(),j))
  else:
    result = j
    result.setNext(recur(i,j.getNext()))
  return result

def mergeSortedLinkedListRecur(ll1,ll2):
  i = ll1.head
  j = ll2.head
  head = recur(i,j)
  resultList = LinkedList()
  resultList.head = head
  print(resultList.detail())
  return resultList

# 递归写法
# 无返回值，更简单，需要将操作目标作为递归函数的变量
def recur2(i,j,tempNode):
  if i is None:
    tempNode.setNext(j)
    return
  if j is None:
    tempNode.setNext(i)
    return
  if i.getValue() <= j.getValue():
    tempNode.setNext(i)
    recur2(i.getNext(),j,i)
  else:
    tempNode.setNext(j)
    recur2(i,j.getNext(),j)

def mergeSortedLinkedListRecur2(ll1,ll2):
  i = ll1.head
  j = ll2.head
  tempNode = Node()
  resultList = LinkedList()
  resultList.head = tempNode
  recur2(i,j,tempNode)
  print(resultList.detail())
  return resultList

#! 删除倒数第 n 个节点
def deleteNodeFromEnd(ll,n):
  size = ll.size()
  count = 1
  cur = ll.head
  prev = None
  while cur and count < size - n:
    count += 1
    prev = cur
    cur = cur.getNext()
  if cur:
    prev.setNext(cur.getNext() or None)
  print(ll.detail())
  return ll

# 双指针
def deleteNodeFromEndPro(ll,n):
  count = 1
  step1 = ll.head
  step2 = None
  cur = ll.head
  while cur and count < n:
    count += 1
    cur = cur.getNext()
  step2 = cur
  while step1.getNext() and step2.getNext():
    prev = step1
    step1 = step1.getNext()
    step2 = step2.getNext()
  prev.setNext(step1.getNext())
  print(ll.detail())
  return ll

#! 求中间节点
def findMiddle(ll):
  step1 = ll.head
  step2 = ll.head
  while step1.getNext() and step2.getNext() and step2.getNext().getNext():
    step1 = step1.getNext()
    step2 = step2.getNext().getNext()
  print(step1.getValue())
  return step1


head1 = Node(0)
head2 = Node(3)
ll1 = LinkedList(head1)
ll2 = LinkedList(head2)
ll1.append(1).append(2).append(7)
ll2.append(4).append(5).append(8)
# mergeList = mergeSortedLinkedListPro(ll2,ll1)
# deleteNodeFromEndPro(mergeList,3)
# findMiddle(mergeList)


# 双向链表定义

class DoublyNode(object):
  def __init__(self,value=None,prev=None,next=None):
    self.value = value
    self.prev = prev
    self.next = next

class DoublyLinkedList(object):
  def __init__(self,head:DoublyNode):
    self.head = head
    self.tail = head
  # 添加节点，与 tail 确立双向指针
  def append(self, node:DoublyNode):
    self.tail.next = node
    node.prev = self.tail
    self.tail = node
    return self
  # 删除特定节点
  def deleteNode(self, node:DoublyNode):
    if node.prev is None:  
      self.head = node.next
    else:
      node.prev.next = node.next
    return self
  # 打印
  def detail(self):
    current = self.head
    while current:
      if current.next is None:
        print(current.value)
      else:
        print(current.value,end="->")
      current = current.next
    return self
  # 没有反转的说法
  def reverse(self):
    pass

# 重新简洁定义单链表
class ListNode:
  def __init__(self, x=None):
    self.val = x
    self.next = None

# 将数组构建为链表
def createLinkedList(arr=[]):
  dummy = cur = ListNode(0)
  for i in arr:
    newNode = ListNode(i)
    cur.next = newNode
    cur = cur.next
  return dummy.next
    
# 打印
def showDetail(head):
  cur = head
  while cur:
    print(cur.val,end='->' if cur.next else '\n') 
    cur = cur.next
    
# 两个链表相加
# 比如 1->2->3 加上 4->5->7 输出 5->7->0->1
def sumLL(l1,l2):
  if l1 is None: return l2
  if l2 is None: return l1
  i = l1
  j = l2
  plusOne = 0
  result = dummy = ListNode(0)
  while i or j:
    # 提前结束的节点值设为 0，帮助简化代码
    iVal = i.val if i is not None else 0
    jVal = j.val if j is not None else 0
    theSum = iVal + jVal + plusOne
    plusOne = 0 if theSum < 10 else 1
    dummy.next = ListNode(theSum % 10)
    dummy = dummy.next
    if i is not None:
      i = i.next
    if j is not None:
      j = j.next
  # 检查最高位
  if plusOne == 1:
    dummy.next = ListNode(1)
  # result 的下一节点就是所求
  return result.next

# 从尾巴删除
def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
  # 处理特殊情况
  if head.next is None and n == 1:
    return None
  count = 1
  i = head
  j = head
  while j and count != n:
    j = j.next
    count += 1
  # 此时 j 指向正数第 n 个节点
  while j.next:
    prev = i
    j = j.next
    i = i.next
  # j 走到尾巴的时候，i 节点就是所求
  if i.next is not None:
    i.val = i.next.val
    i.next = i.next.next
  else:
    prev.next = None
  return head

# 合并 K 个有序链表
import queue
def mergeKSortedLL(lists) -> ListNode:
  dummy = cur = ListNode(0)
  heap = queue.PriorityQueue()
  for l in lists:
    if l:
      heap.put(l.val,l)
  while heap.get():
    temp = heap.get() #取出最小节点
    cur.next = temp
    cur = cur.next
    if temp.next:
      # 最小链表右移一位，继续放入 heap
      temp = temp.next
      heap.put(temp.val,temp)
  return dummy.next

# 每两个节点互换，不允许直接改val
def swapPairs(head):
  if head is None or head.next is None:
    return head
  prev_one = ListNode(0)
  prev = head
  cur = head.next
  dummy = ListNode(0)
  dummy.next = cur
  while cur:
    temp = cur.next
    cur.next = prev
    prev.next = temp
    prev_one.next = cur
    prev_one = prev
    prev = temp
    if prev is not None:
      cur = prev.next
    else:
      cur = None
  return dummy.next

# 递归解法
def swapPairsByRecur(head):
  if head is None or head.next is None:
    return head
  temp = head.next
  head.next = swapPairsByRecur(temp.next)
  temp.next = head
  return temp

# 每 K 个一组进行翻转
# 每次翻转都是经典的单链表翻转，将右边当做已完成翻转的子串
def reverseKGroup(head,k):
  pass

# 删除有序链表中的重复元素，保留一个
def deleteDuplicates(head):
  if not head:
    return head
  cur = head
  while cur.next and cur.next.val == cur.val:
    cur = cur.next
  head.next = deleteDuplicates(cur.next)
  return head

# 删除有序链表中的重复元素，一个都不留
def deleteDuplicatesPro(head):
  if head is None:
    return head
  dummy = prev = ListNode(0)
  cur = head
  prev.next = head
  # 先尝试不重复的，更新 prev 指针
  while cur.next and cur.next.val != cur.val:
    prev = cur
    cur = cur.next
  # 重复的 cur 指针继续右移
  while cur.next and cur.next.val == cur.val:
    cur = cur.next
  # 尾巴区别对待
  prev.next = cur if prev.next == cur else deleteDuplicatesPro(cur.next)
  return dummy.next

# 给定目标值，分隔链表（小的在左边）
def partition(head,x):
  if head is None or head.next is None:
    return head
  dummy = smallPrev  = ListNode(0)
  bigPrev =  solid = None #第一个大于等于x的节点，分界线
  cur = head
  while cur:
    if cur.val < x:
      smallPrev.next = cur
      smallPrev = cur
    else:
      if solid is None:
        solid = cur
      if bigPrev is not None:
        bigPrev.next = cur
      bigPrev = cur
    cur = cur.next
  # 将最后一个小节点指向solid节点
  # 将最后一个大节点指向 None
  bigPrev.next = None
  if smallPrev.val == 0:
    return head
  smallPrev.next = solid
  return dummy.next

# showDetail(partition(createLinkedList([1,4,3,2,5,2]),3))
# 反转部分链表，从 m 到 n
def reverseBetween(head,m,n):
  if head is None or head.next is None or m == n:
    return head
  count = 1
  cur = head
  prev = ListNode()
  start = None
  dummy = preStart = ListNode()
  while cur and count <= n:
    if count == m:
      preStart = prev
      start = cur
    if count == n:
      preStart.next = cur
    next = cur.next
    # 从m到n才需要反转
    if count != 1 and count >= m and count <= n:
      cur.next = prev
    prev = cur
    cur = next
    count += 1
  start.next = cur
  return head if preStart.val is not None else preStart.next
# showDetail(reverseBetween(createLinkedList([1,2,3,4,5]),4,5))

# 利用有序链表构造二叉搜索树 BST
# 定义二叉树节点
class TreeNode(object):
  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None
# 普通解法
def sortedListToBST(head):
  # 将有序链表转换为数组，能够更高效的寻找mid节点
  arr = []
  cur = head
  while cur:
    arr.append(cur.val)
    cur = cur.next
  # 递归构造 BST
  def createBST(arr):
    size = len(arr)
    if size == 0:
      return None
    if size == 1:
      return TreeNode(arr[0])
    mid = size//2
    node = TreeNode(arr[mid])
    node.left = createBST(arr[0:mid])
    node.right = createBST(arr[mid+1:])
    return node
  return createBST(arr)

# TODO 还可以模拟 BST 的中序遍历，构造这棵树
# 因为 BST 的中序遍历结果就是一个升序的序列，相当于反推
def sortedListToBSTPro(head):
  size = 0
  cur = head
  while cur:
    cur = cur.next
    size += 1
  # 模拟中序遍历
  def inOrderIterate(L,R):
    if L > R:
      return None
    mid = (R - L)//2
    leftNode = inOrderIterate(L,mid-1)
    node = TreeNode(head.val)
    head = head.next
    node.left = leftNode
    node.right = inOrderIterate(mid+1,R)
    return node
  return inOrderIterate(0,size-1)

# 复制带随机指针的链表
class RandomNode(object):
  def __init__(self,val):
    self.val = val
    self.next = None
    self.random = None
# 回溯
def copyRandomList(head):
  hashmap = {}
  def traceBack(node):
    if node is None:
      return None
    if node in hashmap:
      return hashmap[node]
    newNode = RandomNode(node.val)
    hashmap[node] = newNode
    newNode.next = traceBack(node.next)
    newNode.random = traceBack(node.random)
    return newNode
  return traceBack(head)

# 找出链表环的起点
def detectCycle(head):
  if head is None:
    return None
  cache = set()
  cur = head
  while cur:
    if cur in cache:
      return cur
    cache.add(cur)
    cur = cur.next
  return None

# 使用双指针 O(1)空间解决，使用了一个巧妙的等式推导:
# fast 指针走过的路程是 slow 指针的两倍，可以得出结论，head 到环起点 tar 的距离等于初次相遇节点 meet 到 tar 的距离
# 因此只需再维护两个指针，分别从 head 和 meet 开始，当他们相遇时就是所求的 tar 节点
def detectCyclePro(head):
  if head is None:
    return None
  slow = head
  fast = head
  meet = None
  # 先找出相遇节点
  while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
      meet = slow
      break
  # 没找到环
  if meet is None:
    return None
  # 维护两个指针，寻找环的起点
  cur = head
  while cur:
    cur = cur.next
    meet = meet.next
    if cur == meet:
      return cur
    
# 重排序 1-2-3-4-5  ->  1-5-2-4-3
# 将链表后半部分翻转即可，然后双指针向中间走
def reorderList(head):
  if head is None or head.next is None:
    return None
  slow = head
  fast = head
  while fast.next and fast.next.next:
    slow = slow.next
    fast = fast.next.next
  mid = slow
  # 注意需要将 mid 节点指向 None，避免死循环
  prev = mid
  slow = mid.next
  mid.next = None
  while slow:
    next = slow.next
    slow.next = prev
    prev = slow
    slow = next
  # prev 就是尾巴节点，维护两个指针，从两边往中间走
  cur = head
  while cur:
    next = cur.next
    cur.next = prev
    next2 = prev.next
    prev.next = next
    prev = next2
    cur = next
  return head

# 链表模拟插入排序
def insertionSortList(head):
  if head is None or head.next is None:
    return head
  prev = head
  cur = head.next
  while cur:
    if cur.val >= prev.val:
      prev = cur
      cur = cur.next
    else:
      # 从 head 开始遍历寻找适合 cur 的位置
      next = cur.next
      s,e = findNodeInSortedList(head,cur)
      if s is None:
        cur.next = e
        head = cur
        prev.next = next
        cur = next
      else:
        s.next = cur
        cur.next = e
        prev.next = next
        cur = next
  return head

# 辅助函数，有序链表中找合适的位置区间 s,e
def findNodeInSortedList(head,node):
  cur = head
  if head.val >= node.val:
    return (None,head)
  while cur.next:
    if cur.val <= node.val and cur.next.val >= node.val:
      return (cur,cur.next)
    cur = cur.next
  return (None,None)

# 时间 O(NlogN)，空间 O(1) 实现链表排序，可以用 bottom-up 的 mergesort
# TODO 可以尝试快排，原地分区
def quickSortLL(head,end=ListNode(None)):
  pass

# 辅助函数，以 head 节点值进行分区，左小右边大
def partitionFirst(head,end):
  if head is None or head.next is None:
    return head
  i = head.next
  j = head.next
  prev = ListNode(None)
  while j and j != end:
    if j.val and head.val and j.val < head.val:
      val = i.val
      i.val = j.val
      j.val = val
      prev = i
      i = i.next
    j = j.next
  # prev 就是分界点
  val = head.val
  head.val = prev.val
  prev.val = val
  return (head,prev)

# showDetail(quickSortLL(createLinkedList([4,2,1,3])))

# 找两个单链表是否相交
def getIntersectionNode(headA,headB):
  if headA is None or headB is None:
    return None
  hashMap = set()
  while headA:
    hashMap.add(headA)
    headA = headA.next
  while headB:
    if headB in hashMap:
      return headB
    headB = headB.next
  return None
# 双指针 O(1) 空间解法
def getIntersectionNodePro(headA,headB):
  if headA is None or headB is None:
    return None
  ptrA = headA
  ptrB = headB
  endA = endB = None
  while True:
    if ptrA == ptrB:
      return ptrA
    if ptrA.next:
      ptrA = ptrA.next
    else:
      endA = ptrA
      if endB and endA != endB:
        return None
      ptrA = headB
    if ptrB.next:
      ptrB = ptrB.next
    else:
      endB = ptrB
      if endA and endB != endA:
        return None
      ptrB = headA

# 递归翻转单链表
def reverseByRecur(head):
  if head is None or head.next is None:
    return head
  right = reverseByRecur(head.next)
  head.next.next = head
  head.next = None
  return right

# odd 位置串联 even 位置，原地重组链表
def oddEvenList(head):
  if head is None or head.next is None or head.next.next is None:
    return head
  odd = head
  even = ListNode(None)
  dummy = head.next
  while odd.next and odd.next.next:
    nextEven = odd.next
    nextOdd = nextEven.next
    even.next = nextEven
    odd.next = nextOdd
    even = nextEven
    odd = nextOdd
  # 偶数尾巴特殊处理
  even.next = odd.next
  odd.next = dummy
  return head

# 两链表从高位开始相加，不允许翻转链表
# 利用 stack 来实现某种翻转操作
def addTwoNumsNoReverse(l1,l2):
  if l1 is None:
    return l2
  if l2 is None:
    return l1
  stack1 = []
  stack2 = []
  i = l1
  j = l2
  while i:
    stack1.append(i.val)
    i = i.next
  while j:
    stack2.append(j.val)
    j = j.next
  plusOne = 0
  tail = None
  while len(stack1) or len(stack2):
    val1 = stack1.pop().val if len(stack1) else 0
    val2 = stack2.pop().val if len(stack2) else 0
    theSum = val1 + val2 + plusOne
    plusOne = theSum // 10
    remainder = theSum % 10
    node = ListNode(remainder)
    node.next = tail
    tail = node
  if plusOne == 1:
    head = ListNode(1)
    head.next = tail
    return head
  return tail

# 拍平双向链表
# 递归解法，假设 child 已经完成
def flatten(head):
  if head is None:
    return head
  cur = head
  while cur:
    if cur.child:
      flattenedChild = flatten(cur.child)
      end = findEnd(flattenedChild)
      next = cur.next
      cur.next = flattenedChild
      flattenedChild.prev = cur
      end.next = next
      if next:
        next.prev = end
      cur.child = None
    cur = cur.next
  return head

# 辅助函数
def findEnd(head):
  if head is None:
    return head
  cur = head
  while cur.next:
    cur = cur.next
  return cur

# stack 解法，经典的 BFS？
# 遇到 child 就将 next 压入栈，然后 next 指针往 child 继续遍历
def flattenByStack(head):
  if head is None:
    return head
  stack = []
  cur = head
  while cur:
    if cur.child:
      if cur.next:
        stack.append(cur.next)
      cur.next = cur.child
      cur.child.prev = cur
      cur.child = None
    # child 已经遍历完成，开始从 stack 中取出原来的 next 子串
    if cur.next is None and len(stack):
      tail = stack.pop()
      cur.next = tail
      tail.prev = cur
    cur = cur.next
  return head
  
# 切分 LL 为 k 份
def splitListToParts(root,k):
  cur = root
  size = 1
  while cur.next:
    cur = cur.next
    size += 1
  avgSize = size // k
  remainder = size % k
  result = []
  while k:
    if remainder:
      root, part = handleRoot(root,avgSize+1)
      result.append(part)
      remainder -= 1
    else:
      root, part = handleRoot(root,avgSize)
      result.append(part)
    k -= 1
  return result

# 辅助函数，切一小片 root 并返回新 root
def handleRoot(root,size):
  if root is None or size == 0:
    return (None,None)
  count = 1
  cur = root
  while cur and count < size:
    cur = cur.next
    count += 1
  next = cur.next
  cur.next = None
  return (next,root)

# 给定一个链表和一个由部分值组成的数组，求链表被分隔为几部分
# 此题的核心不是求分界点，而是大量判断 val in G，因此将 G 转换为 set 是关键
def numComponents(head,G=[]):
  if head is None or head.next is None:
    return 1
  setG = set(G)
  result = 0
  while head:
    if head in setG and(head.next not in setG):
      result += 1
  return result

# 找出最接近的比 cur 大的节点，以 idex 为下标构造数组
# 维护一个 stack，如果栈顶大于当前节点，将 (idx,val) 压入栈，反之出栈，给 res 数组加一个元素
def nextLargerNodes(head):
  if head is None:
    return []
  stack = []
  res = []
  while head:
    # 栈顶比当前元素小的，需要循环检查
    while len(stack) and stack[-1][1] < head.val:
      idx = stack.pop()[0]
      res[idx] = head.val
    stack.append((len(res),head.val))
    # 每检查一个元素，往 res 添加一个空元素进行占位
    res.append(0)
    head = head.next
  # 如果 stack 中还有元素，说明没有比他们大的节点
  while len(stack):
    idx = stack.pop()[0]
    res[idx] = 0
  return res