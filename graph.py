# 在内存中，图一般采用邻接（链）表存储，与哈希表非常相似
# 维护 size v 的链表数组，下标是顶点
# 每个链表存储该顶点所连接的其他顶点
# 有向图需要额外维护一个链表数组

from collections import deque
class Graph(object):
  # 给出顶点 s，adj[s] 就表示与它相连的顶点们
  def __init__(self, v):
    self.v = v
    self.adj = [None] * v
    for i in range(v):
      # 为简化代码，这里使用 deque 模拟链表/红黑树/跳表
      self.adj[i] = deque()
  # 将两个顶点相连，此处为无向图
  def addEdge(self, s, t):
    self.adj[s].append(t)
    self.adj[t].append(s)
  
  #! BFS
  # 与二叉树的按层遍历类似，使用 queue
  # 额外维护一个哈希表，防止重复遍历
  # BFS 找到的是最短路径，DFS 则不是
  # 注意，如果不指定起点，且默认起点无法连通到所有其他顶点，则需要改造一下代码，遍历尝试从所有起点开始 BFS 寻找
  def bfs(self, s, t):
    queue = deque([s])
    visited = {}
    res = []
    while queue:
      node = queue.popleft()
      # 换成检查 adj 也行，只需将起点 s 标记 visited
      if node in visited:
        continue
      visited[node] = True
      res.append(node)
      if node == t:
        print(res)
        return res
      for i in self.adj[node]:
        queue.append(i)
        

  #! DFS
  # 找到的只是路径之一，并不是最短路径，需要额外的 found 指针，在找到目标时停止其他路径的查找
  def dfs(self, s, t):
    res = []
    visited = {}
    found = False
    def recur(s,t):
      nonlocal res,visited,found,self
      if found:
        return
      if s == t:
        res.append(s)
        found = True
        return
      res.append(s)
      visited[s] = True
      for i in self.adj[s]:
        # 放在上面检查当前节点也行
        if i not in visited:
          recur(i,t)
    recur(s,t)
    print(res)
    return res

graph = Graph(5)
graph.addEdge(0,1)
graph.addEdge(1,4)
graph.addEdge(0,3)
graph.addEdge(1,2)
graph.addEdge(2,3)
graph.addEdge(4,3)

graph.bfs(0,2)
graph.dfs(0,2)


# Definition for a Node.
class Node:
  def __init__(self, val, neighbors):
    self.val = val
    self.neighbors = neighbors

class Solution:
  # Clone Graph 深度复制
  def cloneGraph(self, node: 'Node') -> 'Node':
    # BFS
    pass

  # TODO Course Schedule 课程表，总共 n 个课程，他们之间两两可能存在依赖
  # [1,0] 表示课程 1 依赖 0，提供一个独特的二维数组描述有向图
  # 需要先转换为常规的邻接表，问题等价于探测有向图中是否存在环
  def canFinish(self, numCourses: int, prerequisites) -> bool:
    pass