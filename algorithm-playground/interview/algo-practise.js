// 以下为常规算法大类

// 给定一组天气，求另一数组，记录下一天更热的等待期
// 考察 stack、monostack
// stack 可以记录 reference，也可以只记录下标
function dayTempreture(arr) {
  const n = arr.length,
    stack = [0];
  let res = Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    while (stack.length && arr[i] > arr[stack[stack.length - 1]]) {
      const low = stack.pop();
      res[low] = i - low;
    }
    stack.push(i);
  }
}

// 给定一棵二叉树，求有多少个子树，其所有元素拥有相同数字
// !考察递归和二叉树的 post-order traverse
function countUnivalSubtrees(root) {
  let res = 0;
  // 辅助函数，判断是否
  function recur(root) {
    if (!root) return true;
    const l = recur(root.left),
      r = recur(root.right);
    if (!l || !r) return false;
    if (root.left && root.left.val !== root.val) return false;
    if (root.right && root.right.val !== root.val) return false;
    // 同时操作外部变量
    res += 1;
    return true;
  }
  recur(root);
  return res;
}

// 给定一棵 BST，求第 k 个最小的元素
var kthSmallest = function(root, k) {
  // bst 中序遍历是有序的，只需要检查一下第几个就行
  let res = null;
  function recur(root) {
    if (!root) return;
    recur(root.left);
    k -= 1;
    if (k === 0) {
      res = root.val;
      return;
    }
    recur(root.right);
  }
  recur(root, k);
  return res;
};

// !寻找两个节点的最近父节点
var lowestCommonAncestor = function(root, p, q) {
  // 关键是辅助函数，判断当前树是否含有目标节点，返回值 0 或 1
  // 如果 left、root、right 三者中有两个，那么 root 就是所求
  let res = root;
  // 前序遍历
  function recur(root) {
    if (!root) return 0;
    let count = 0;
    if (root.val === p.val || root.val === q.val) {
      count += 1;
    }
    const left = recur(root.left);
    const right = recur(root.right);
    if (count + left + right === 2) {
      res = root;
      return;
    }
    // tricky part: 只能返回 0 或 1
    return left || count || right;
  }
  recur(root);
  return res;
};

// 实现一个循环队列
// ! 核心是求余获得下一个位置，以及永远在 tail 后边留一个空位
// ? 为什么要留一个空位呢： 否则队列满了之后，继续调用一次 enqueue 会替换队列中的元素
// ? 为什么要取余数获得新位置：为了避免在边界位置的时候还要单独判断
// 教科书式的写法是没有 curSize 辅助变量的，只用 (tail + 1) % size === head 判满，以及 head === tail 判空
// 双端循环队列实现类似，只不过退位的时候需要 (head - 1 + maxSize) % maxSize 获得新位置
class LoopQueue {
  constructor(size) {
    this.arr = Array(size + 1).fill(null);
    this.maxSize = size + 1;
    this.curSize = 0;
    this.head = 0;
    this.tail = 0;
  }
  enqueue(ele) {
    // 不能简化取值，必须写全 this.xxx
    if (this.curSize + 1 === this.maxSize) {
      console.warn("The queue is full.");
      return;
    }
    this.arr[this.tail] = ele;
    this.curSize += 1;
    this.tail = (this.tail + 1) % this.maxSize;
    return this;
  }
  dequeue() {
    if (this.curSize === 0) {
      console.warn("The queue is empty.");
      return;
    }
    const res = this.arr[this.head];
    this.head = (this.head + 1) % this.maxSize;
    this.curSize -= 1;
    return res;
  }
  // !获取队列尾的值，需要注意
  getLast() {
    return this.arr[(this.tail - 1 + this.maxSize) % this.maxSize];
  }
}

// 实现一组数据的全排列
// 如果求数量，也就是阶乘，那可以归属于 DP
// 如果求所有排列本身，那属于回溯
function permutation(arr) {
  const res = [];
  const n = arr.length;
  // !如果求部分排列，比如 n 取 m 排列，这里结束条件改成 i === m 即可
  function recur(i, temp) {
    if (i === n) {
      res.push(temp);
      return;
    }
    for (let j = 0; j < n; j++) {
      // 这里也可以用 arr 的下标作为唯一标识
      if (!temp.includes(arr[j])) {
        const newTemp = temp.concat();
        newTemp.push(arr[j]);
        recur(i + 1, newTemp);
      }
    }
  }
  recur(0, []);
  return res;
}

// 求 n 取 m 的所有组合
function combination(n, k) {
  const res = [];
  function recur(i, start, temp) {
    if (i === k) {
      res.push(temp);
      return;
    }
    // !在上述排列的写法基础上，巧妙的利用 start 缩小可选的 j 范围
    // start 跟 i 其实可以合并，这里分开写是为了表明它们各自的作用
    for (let j = start; j <= n; j++) {
      const newTemp = temp.concat();
      newTemp.push(j);
      recur(i + 1, j + 1, newTemp);
    }
  }
  recur(0, 1, []);
  return res;
}

// 二分查找变种：大于等于给定值的第一个元素
function bsFirstGreater(arr, k) {
  let low = 0,
    high = arr.length - 1;
  if (arr[0] >= k) return arr[0];
  while (low <= high) {
    const mid = low + Math.floor((high - low) / 2);
    const midVal = arr[mid];
    if (midVal < k) {
      low = mid + 1;
    } else {
      // 看看左边是不是比 k 小，是的话答案就 mid 了
      if (arr[mid - 1] < k) {
        return arr[mid];
      } else {
        high = mid - 1;
      }
    }
  }
  return null;
}

// 插入排序：将左侧排好序，一路右移，找位置插入
function insertSort(arr) {
  const n = arr.length;
  for (let i = 1; i < n; i++) {
    const temp = arr[i];
    let j = i - 1;
    while (j > -1 && arr[j] > temp) {
      // 主要是直接赋值 arr[j+1] = arr[j]
      arr[j + 1] = arr[j];
      j--;
    }
    arr[j + 1] = temp;
  }
  return arr;
}

// 冒泡排序：每次将最大值交换到右边
function bubbleSort(arr) {
  const n = arr.length;
  let isSwap = true;
  for (let k = 0; k < n; k++) {
    if (isSwap) {
      for (let i = 0; i < n - k - 1; i++) {
        isSwap = false;
        if (arr[i + 1] < arr[i]) {
          [arr[i + 1], arr[i]] = [arr[i], arr[i + 1]];
          isSwap = true;
        }
      }
    }
  }
  return arr;
}

// Trie 树
class TrieNode {
  constructor(val = null) {
    this.val = val;
    this.children = Array(26).fill(null);
  }
}
class Trie {
  constructor() {
    this.root = new TrieNode("/");
  }
  add(word) {
    let node = this.root;
    for (const w of word) {
      const idx = w.charCodeAt() - 97;
      if (!node.children[idx]) node.children[idx] = new TrieNode(w);
      node = node.children[idx];
    }
    node.isWord = true;
  }
  checkExists(word) {
    let node = this.root;
    for (const w of word) {
      const idx = w.charCodeAt() - 97;
      if (!node.children[idx]) return false;
      node = node.children[idx];
    }
    return node.isWord;
  }
  findPrefixWords(prefix) {
    let node = this.root;
    for (const w of prefix) {
      const idx = w.charCodeAt() - 97;
      if (!node.children[idx]) return [];
      node = node.children[idx];
    }
    // 回溯找出所有 isWord 的节点
    const res = [];
    if (node.isWord) res.push(prefix);
    function recur(node, tempStr) {
      if (!node) return;
      // children 就是你的子问题
      for (const c of node.children) {
        if (c) {
          tempStr += c.val;
          if (c.isWord) res.push(tempStr);
          recur(c, tempStr);
        }
      }
    }
    recur(node, prefix);
    return res;
  }
}

// !二叉查找树某个节点的后继节点
var inorderSuccessor = function(root, p) {
  // 完整中序遍历 + prev 指针解法，复杂度是 O(n)
  // 下面这种最优解法，复杂度 O(h) 也就是 O(logn)
  // 如果目标 p 有右树，那么从右树种找到最小值即可
  // 如果目标 p 没有右树，从 root 开始找，不断往 p 靠近，每次大于 p 的时候更新一下 res
  // 最后找到 p 的时候停止，这时 res 刚好是最近一次大于 p 的
  let res = null;
  if (p.right) {
    // 如果有右树，找到右树的最小值即可
    p = p.right;
    while (p.left) {
      p = p.left;
    }
    return p;
  }
  // 没有右树，不断向 p 逼近
  while (root) {
    if (root.val > p.val) {
      res = root;
      root = root.left;
    } else if (root.val < p.val) {
      root = root.right;
    } else {
      // 找到 p 了
      break;
    }
  }
  return res;
};

// 另一种解法：如果只给 p 节点，但每个节点知道自己的 parent
// 如果有右树，也是找右边最小节点
// !如果没有右树，目标肯定在上面，只需沿 parent 向上找，第一个 node.parent.left === node 的节点
// node.parent 就是目标，因为往右上方是递增的，node.parent 就是刚好比 p 大

// 找前驱节点
// 跟上面类似，也是分两种情况
// 如果有左树，则找到左树的最大子节点即可
// !如果没有左树，从 root 开始逼近，每次 root 小于 p，就赋值 res = root
// 最后找到 p 的时候，最近一次 res 就是目标节点
function inorderPrevNode(root, p) {
  if (p.left) {
    p = p.left;
    while (p.right) {
      p = p.right;
    }
    return p;
  }
  let res = null;
  while (root) {
    if (root.val < p.val) {
      res = root;
      root = root.right;
    } else if (root.val > p.val) {
      root = root.left;
    } else {
      break;
    }
  }
  return res;
}

// 图的表示，通常用邻接表的方式，优化点：将节点的 neighbors 换成链表、红黑树、哈希表等
// 如果用邻接矩阵表示，按照已知顶点数初始化二维矩阵，this.graph = Array.from(Array(v),() => new Array(v))
// 然后 graph(i,j) 表示 i 连接 j
class GraphNode {
  constructor(val) {
    this.val = val;
    this.neighbors = []; // !这里用数组模拟链表，也属于邻接表表示
  }
}
// 一般来说，最常见的是根据已知顶点数量初始化
class Graph {
  constructor(val) {
    // 根据实际需要，也可以支持批量顶点初始化
    this.vertex = new GraphNode(val);
    this.allNodes = [this.vertex];
    this.size = 1;
  }
  addNode(val) {
    // 只允许按照数字递增关系添加，比如上一个是 4，这次只能添加 5
    this.allNodes.push(new GraphNode(val));
    this.size++;
  }
  linkNode(sVal, tVal) {
    // 这里要求顶点的 val 就是它的下标
    this.allNodes[sVal].neighbors.push(t);
    this.allNodes[tVal].neighbors.push(s);
  }
  bfs(s, t) {
    let visited = {},
      queue = [s];
    while (queue) {
      const node = queue.shift();
      visited[node] = true;
      if (s === t) return true;
      for (const n of node.neighbors) {
        if (!(n in visited)) {
          queue.push(n);
        }
      }
    }
    return false;
  }
  dfs(s, t) {
    // 通常还需要 isFound 指针，找到后立即停止
    const visited = {};
    function recur(node, t) {
      if (!node) return false;
      if (node === t) {
        return true;
      }
      visited[node] = true;
      let curRes = false;
      for (const n of node.neighbors) {
        if (!(n in visited)) {
          if (recur(n, t)) {
            curRes = true;
            break;
          }
        }
      }
      return curRes;
    }
    return recur(s, t);
  }
}

// 一个二维矩阵由 0 和 1 组成，相连的 1 称为岛屿，求岛屿数量
var numIslands = function(grid) {
  // 直觉是回溯思想、DFS
  if (!grid.length || !grid[0].length) return 0;
  const m = grid.length,
    n = grid[0].length;
  const visited = Array.from(Array(m), () => new Array(n));
  let count = 0;

  function recur(i, j) {
    // 越界，停止
    if (i < 0 || i >= m || j < 0 || j >= n) return;
    if (grid[i][j] === "0") return;
    // 已访问过，停止
    if (visited[i][j]) return;
    visited[i][j] = true;
    recur(i + 1, j);
    recur(i - 1, j);
    recur(i, j + 1);
    recur(i, j - 1);
  }
  // 尝试全部走一次
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      // !tricky part: recur 函数本身只负责走当前岛屿，以及填充 visited，岛屿的数量只能通过这里统计
      // 如果是统计岛屿面积，那么就需要在 recur 函数里进行
      if (grid[i][j] === "1" && !visited[i][j]) count += 1;
      recur(i, j);
    }
  }
  return count;
};
// 进阶：走过的时候将 1 污染为 0，这样就不需要 visited 缓存了。

// 回溯思想经典题目
// 八皇后问题：一个 8*8 的棋盘，放入 8 个皇后，要求横、竖、对角线上不能有其他皇后
// 求所有可行的组合，正确答案是 92 种
function the8Queen() {
  const res = [];
  // 分 8 步走，不合格的提前淘汰
  function recur(i, tempRes) {
    if (i >= 8) {
      res.push(tempRes);
      return;
    }
    for (let j = 0; j < 8; j++) {
      // 判断 (i,j) 这一格能不能走
      if (isValid(tempRes, i, j)) {
        const copyRes = tempRes.concat();
        copyRes.push(j);
        recur(i + 1, copyRes);
      }
    }
  }
  // 辅助函数，判断当前 arr 数组是否满足竖向、双对角线条件
  // arr 存储当前的走法，下标表示 row，值表示 col
  // !与数独的判断类似，我们希望在循环里，一次判断 3 种情况
  // 这种写法是最简单优雅的，只需要判断最后这个位置 (i,j) 是否会与前面冲突
  function isValid(arr, i, j) {
    let leftUp = j - 1,
      rightUp = j + 1;
    for (let k = i; k > 0; k--) {
      // !同时判断竖向和双对角线上是否存在冲突，注意只能往上判断
      if (arr[k - 1] === j) return false;
      if (arr[k - 1] === leftUp || arr[k - 1] === rightUp) return false;
      leftUp--;
      rightUp++;
    }
    return true;
  }
  recur(0, []);
  return res;
}

// 0-1 背包问题：给定一组物品重量 weights，背包的最大负重是 w，求最大负重组合。
// 经典解法是 DP，当然回溯也能解
function getMaxWeight(weights, w) {
  let curMaxW = 0,
    res = null,
    n = weights.length;
  const cache = {};
  function recur(i, curW, curArr) {
    if (i === n) {
      if (curW > curMaxW) res = curArr;
      return;
    }
    // 到当前这一步，重量相等的，没必要都往下算了
    if (`step ${i}, weight is ${curW}` in cache) return;
    cache[`step ${i}, weight is ${curW}`] = true;
    // 不装
    recur(i + 1, curW, curArr);
    // 装
    if (curW + weights[i] < w) {
      const copyArr = curArr.concat();
      copyArr.push(weights[i]);
      recur(i + 1, curW + weights[i], copyArr);
    }
  }
  recur(0, 0, []);
  return res;
}

// DP 解法：适合求最大重量，如果求具体组合的话，需要倒推，稍微麻烦一点
// 状态转移表覆盖，去重
function getMaxWeightDP(weights, w) {
  const n = weights.length;
  const dp = Array.from(Array(n), () => new Array(w));
  // 初始化
  dp[0][0] = true;
  if (weights[0] < w) dp[0][weights[0]] = true;
  // 填充
  for (let i = 1; i < n; i++) {
    // 不装
    for (let j = 0; j < w; j++) {
      if (dp[i - 1][j] !== undefined) dp[i][j] = true;
    }
    // 装
    for (let j = 0; j < w - weights[i]; j++) {
      if (dp[i - 1][j] !== undefined) dp[i][j + weights[i]] = true;
    }
  }
  // 最右下角就是答案
  for (let i = w - 1; i > -1; i--) {
    if (dp[n - 1][i] === true) return i;
  }
}

// !求一个数组的逆序度 inversions
// 分治思想
// 使用改造后的归并排序：逆序度 = 左边逆序度 + 右边逆序度 + 左边对右边的逆序度
// 以前是返回排好序的数组，现在加一个额外逆序度即可
function inversions(arr) {
  // 改造归并函数，现在返回两个东西
  function mergeSort(arr, left, right) {
    if (left === right) return [0, [arr[left]]];
    const mid = left + Math.floor((right - left) / 2);
    let [leftCount, leftArr] = mergeSort(arr, left, mid);
    let [rightCount, rightArr] = mergeSort(arr, mid + 1, right);
    let [interCount, curArr] = merge(leftArr, rightArr);
    return [leftCount + rightCount + interCount, curArr];
  }
  // 改造 merge 函数
  // 合并两个有序序列，过程中计算此次的逆序度，返回两个元素
  function merge(leftArr, rightArr) {
    let count = 0,
      l1 = leftArr.length,
      l2 = rightArr.length,
      i = 0,
      j = 0;
    let temp = [];
    while (i < l1 && j < l2) {
      if (leftArr[i] <= rightArr[j]) {
        temp.push(leftArr[i]);
        i++;
      } else {
        temp.push(rightArr[j]);
        count += l1 - i; // 因为 i 到其尾巴之间的元素都比 j 大
        j++;
      }
    }
    while (i < l1) {
      temp.push(leftArr[i]);
      i++;
    }
    while (j < l2) {
      temp.push(rightArr[j]);
      j++;
    }
    return [count, temp];
  }
  return mergeSort(arr, 0, arr.length - 1)[0];
}
