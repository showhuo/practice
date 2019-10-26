# 字典树
# 常用于模糊查找，共同前缀的字符串
# 精确查找不如使用哈希表、红黑树、跳表等
# 具体应用有输入联想、自动补齐等

from collections import deque

class Node():
  def __init__(self, val=None):
    self.val = val
    # 追求极致性能的话，这里改成 size K 的数组，K 是字符集中不同的字母个数
    self.children = {}
    self.isEnd = False

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

# Your Trie object will be instantiated and called as such:
obj = Trie()
obj.insert('abc')
obj.insert('def')
obj.insert('mmm')
# obj.bfs()
# obj.dfs()
print(obj.searchWithDot('...c'))


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