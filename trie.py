# 字典树
# 常用于模糊查找，共同前缀的字符串
# 精确查找不如使用哈希表、红黑树、跳表等
# 具体应用有输入联想、自动补齐等

from collections import deque
from collections import Counter

class Node():
  def __init__(self, val=None):
    self.val = val
    # 追求极致性能的话，这里改成 size K 的数组，K 是字符集中不同的字母个数
    self.children = {}
    self.isEnd = False
    # 为后续问题服务
    self.idx = -1

class Trie:

  def __init__(self):
    """
    Initialize your data structure here.
    """
    self.root = Node('/')

  def insert(self, word: str) -> None:
    """
    Inserts a word into the trie.
    """
    node = self.root
    for c in word:
      if c not in node.children:
        node.children[c] = Node(c)
      node = node.children[c]
    node.isEnd = True
    
  def search(self, word: str) -> bool:
    """
    Returns if the word is in the trie.
    """
    node = self.root
    for c in word:
      if c not in node.children:
        return False
      node = node.children[c]
    if not node.isEnd:
      return False
    return True

  def startsWith(self, prefix: str) -> bool:
    """
    Returns if there is any word in the trie that starts with the given prefix.
    """
    node = self.root
    for c in prefix:
      if c not in node.children:
        return False
      node = node.children[c]
    return True

  # 支持正则符号 . 
  # 回溯思想，防止遗漏
  def searchWithDot(self, word: str) -> bool:
    """
    . 可以表示任意字符
    """
    def backTracking(node,word):
      if word and not node:
        return False
      if node and not word:
        return node.isEnd
      if word[0] != '.':
        if word[0] not in node.children:
          return False
        return backTracking(node.children[word[0]],word[1:])
      else:
        for c in node.children.values():
          # 任意一个匹配就行
          if backTracking(c,word[1:]):
            return True
        return False
    return backTracking(self.root,word)

  # 通常没人会对 Trie 进行 bfs 遍历
  def bfs(self):
    queue = deque([self.root])
    while queue:
      head = queue.popleft()
      print(head.val)
      for c in head.children.values():
        if c:
          queue.append(c)
  
  def dfs(self):
    def recur(node):
      if not node:
        return
      print(node.val)
      for c in node.children.values():
        recur(c)
    recur(self.root)


# 给定 n*n 的二维字符矩阵，以及一个 words list，字符在矩阵中只能横竖连续组合，找出 list 中能如此构造的字符串
# 回溯思想，通过 words 构造一个 Trie，将矩阵中的每个点作为起点，在 Trie 中寻找满足的单词
# 难度：hard
def findWords(board,words):
  res = []
  trie = Trie()
  for w in words:
    trie.insert(w)
  node = trie.root
  for i in range(len(board)):
    for j in range(len(board[0])):
      backTracking(board,node,i,j,'',res)
  return res

# 辅助函数，回溯
# node 表示字典树节点，(i,j) 表示矩阵元素，二者是联动关系
def backTracking(board,node,i,j,path,res):
  if node.isEnd:
    # 还可以继续往下，因此不 return
    res.append(path)
    node.isEnd = False
  if i < 0 or i > len(board)-1 or j < 0 or j > len(board[0])-1:
    return
  tmp = board[i][j]
  if tmp not in node.children:
    return
  node = node.children[tmp]
  # !防止往回走，因此用哨兵污染当前(i,j)位置，这样往回走就走不通
  board[i][j] = '#'
  # path 要累加当前字符
  backTracking(board,node,i+1,j,path+tmp,res)
  backTracking(board,node,i-1,j,path+tmp,res)
  backTracking(board,node,i,j+1,path+tmp,res)
  backTracking(board,node,i,j-1,path+tmp,res)
  # 复原哨兵位置，否则会影响同级的回溯
  board[i][j] = tmp

# 给定一组 unique words，找出下标组合(i,j)使得 words[i]+words[j] 构成回型字符串（palindrome）
# 难度：hard
# TODO
def palindromePairs(words):
  res = []
  idxHashmap = {}
  for i in range(len(words)):
    idxHashmap[words[i]] = i
  for k in range(len(words)):
    w = words[k]
    for j in range(len(w)+1):
      pref = w[:j]
      suffix = w[j:]
      if isPalindrome(pref) and suffix[::-1] in idxHashmap:
        res.append((idxHashmap[suffix[::-1]],j))
      if isPalindrome(suffix) and pref[::-1] in idxHashmap:
        res.append((j,idxHashmap[pref[::-1]]))
  return res

# 辅助函数，检查是否回型字符串
def isPalindrome(s:str):
  return s == s[::-1]

# 有且仅有一个错别字的匹配
class MagicDictionary:
  def __init__(self):
    self.trie = Trie()
      
  def buildDict(self, dicts) -> None:
    for s in dicts:
      self.trie.insert(s)

  def search(self, word: str) -> bool:
    """
    Returns if there is any word in the trie that equals to the given word after modifying exactly one character
    """
    res = False
    counter = 0
    # 辅助函数，回溯
    def traceBack(node,word,counter):
      nonlocal res
      # 唯一满足条件
      if not word and node.isEnd and counter == 1:
        res = True
        return
      # 常规终止
      if not node or not word:
        return
      w = word[0]
      # 我们不关心精确匹配，每一层都全部扫描，只在字符不同时，给当前 counter 加1
      for k in node.children:
        if k != w:
          traceBack(node.children[k],word[1:],counter+1)
        else:
          traceBack(node.children[k],word[1:],counter)
    # 执行
    traceBack(self.trie.root,word,counter)
    return res

# 求共同前缀的key，对应 value 的 sum
# 比检查前缀存在复杂一点，需要回溯找出共同前缀的所有节点
class MapSum:
  def __init__(self):
    """
    Initialize your data structure here.
    """
    self.root = Node('root')

  def insert(self, key: str, val: int) -> None:
    # 除了标记 isEnd，需要额外挂载 val 到节点上
    node = self.root
    for c in key:
      if c not in node.children:
        node.children[c] = Node(c)
      node = node.children[c]
    node.extraVal = val

  def sum(self, prefix: str) -> int:
    res = 0
    node = self.root
    for c in prefix:
      if c not in node.children:
        return res
      node = node.children[c]
    # 开始回溯
    def recur(node):
      nonlocal res
      if not node:
        return
      if node.extraVal:
        res += node.extraVal
      for n in node.children.values():
        recur(n)
    recur(node)
    return res

class StrNode():
  def __init__(self, v=None):
    self.v = v
    self.children = [None]*26 # 只支持26个小写字母，可以辅助字母排序
    self.word = None # 记录完整单词，同时起到标记 isWord 的作用

# 找出 Top K frequent 单词，频率相同的按字母排序（伪top k问题）
# 除了大顶堆，我们也可以用 bucket + Trie 解决
# !frequency 相同的单词放入同个 bucket，字母排序根据 Trie Node 的 children 有序数组实现
def topKFrequent(self, words, k: int):
  # 统计次数
  # countMap = Counter(words)
  pass

# 找最长连续单词
# 这种 word 标记 + 回溯的解法也可以打印出指定前缀的所有单词
def longestWord(words) -> str:
  # 构造一颗 trie，查找时求最长连续 word 标记
  trie = StrNode('/')
  for word in words:
    # !注意 node 指针必须在这个位置重置
    node = trie
    for w in word:
      idx = ord(w) - ord('a')
      if not node.children[idx]:
        node.children[idx] = StrNode(w)
      node = node.children[idx]
    node.word = word
  # 开始查找
  node = trie
  res = ''
  # 辅助函数，回溯
  def recur(node):
    nonlocal res
    # 一旦不连续，放弃当前分支
    if not node or not node.word:
      return
    # 长度更新
    if len(node.word) > len(res):
      res = node.word
    for n in node.children:
      recur(n)
  # 从第一层开始，防止 root.word 报错
  for n in node.children:
    recur(n)
  return res

# 给定一个 words 数组，下标 i 的权重 i，求满足 (prefix, suffix) 条件的权重最大的单词
class WordFilter:
  def __init__(self, words):
    self.words = words
    # fPro 解法：构造一颗大 trie，应对频繁查找
    # !Trie 的好处在于，使用大量内存空间换取效率
    # 这里我们构造一颗“超大”的 trie：对每一个 word，找出所有的 suffix，用 suffix + "#" + word 插入树里，同时让节点挂载更高的权重
    self.trie = Node()
    for k in range(len(words)):
      word = words[k]
      for i in range(len(word)+1):
        node = self.trie
        tempStr = word[i:] + '#' + word
        for w in tempStr:
          if w not in node.children:
            node.children[w] = Node(w)
          node = node.children[w]
          node.idx = k
        # 辅助可视化
        # node.word = tempStr
        # print(node.word)


  def f(self, prefix: str, suffix: str) -> int:
    res = -1
    if not prefix and not suffix:
      return len(self.words) - 1
    for i in range(len(self.words)-1,-1,-1):
      word = self.words[i]
      # 这种 BF 解法是正确的，不过只适合一次性查找，注意 suffix 必须指定起始查找位置
      if word.find(prefix) == 0 and word.find(suffix,len(word)-len(suffix)) == len(word)  - len(suffix):
        res = i
        break
    return res

  # 查找时，找到 suffix + "#" + prefix 即可
  def fPro(self, prefix: str, suffix: str) -> int:
    if not prefix and not suffix:
      return len(self.words) - 1
    node = self.trie
    theStr = suffix + '#' + prefix
    for w in theStr:
      if w not in node.children:
        return -1
      node = node.children[w]
    return node.idx

obj = WordFilter(['apple'])
print(obj.f('a','e'))