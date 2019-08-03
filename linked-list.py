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
    def __init__(self, x):
        self.val = x
        self.next = None

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
def reverseKGroup(head,k):
  pass