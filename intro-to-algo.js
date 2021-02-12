// 2.1-2 nonincreasing Insert-Sort
// 两层迭代，大迭代从第2位往右边走，小迭代从j-1往左边走，小迭代只跟 array[j] 比较大小
// 因为 array[0...j-1] 是已经排好序的，只要找到属于 array[j] 的位置，其他人右移一位
function nonIncreasingInsertSort(array) {
  const n = array.length;
  for (let j = 1; j < array.length; j++) {
    const element = array[j];
    let i = j;
    while (i > 0 && array[i - 1] <= element) {
      array[i] = array[i - 1];
      i--;
    }
    // 小循环终止时，给 i 位赋值
    array[i] = element;
  }
  return array;
}

// console.log(nonIncreasingInsertSort([1, 2, 2, 3, 4, 5]));

// 2.1-3 循环不变式辅助证明线性搜索的正确性
// 三要素：循环开始前，条件成立
// 当次迭代前如果条件成立，迭代后条件肯定也成立
// 迭代结束时，有数据证明结论正确
// 此处的不变式可以是 i = index || null, index < length
function searchIdx(array, v) {
  for (let index = 0; index < array.length; index++) {
    if (v === array[index]) return index;
  }
  return null;
}

// 2.1-4 两个数组相加
function add2Array(A, B) {
  const n = A.length = B.length;
  const res = Array(n + 1).fill(null);
  let extra = 0;
  for (let index = n - 1; index >= 0; index--) {
    const curSum = A[index] + B[index] + extra;
    const curNum = curSum % 10;
    // 如果 A B 长度不相等，断点处的进位就有可能是 2
    extra = Math.floor(curSum / 10);
    res[index + 1] = curNum;
  }
  res[0] = extra;
  return res;
}

// 2.2-2 Selection sort 选择排序，每次从剩余的选取最小的放到左侧
// 剩余的总是越来越少，左侧是有序的，只需要放到它们的最后
// 循环不变式：左侧是有序的
function selectSort(A) {
  for (let j = 0; j < A.length; j++) {
    // 找到右侧最小的位置，与 j 位置交换
    const minIdx = findMinIdx(j, A);  // 执行 n 次，每次复杂度约 n/2，因此该算法复杂度 O(n*n)
    [A[j], A[minIdx]] = [A[minIdx], A[j]];
  }
  return A;
}

function findMinIdx(start, A) {
  let minIdx = start;
  for (let index = start; index < A.length; index++) {
    if (A[index] < A[minIdx]) minIdx = index;
  }
  return minIdx;
}
// console.log(selectSort([7, 2, 2, 9, 4, 5]));

// 2.3-4 插入排序的递归写法
// 子问题：将 A[1..n-1] 排序，将 A[n] 插入到有序数组中，获得答案。
function InsertSortRecur(A, i) {
  if (i === 0) return;

  InsertSortRecur(A, i - 1);
  let j = i - 1;
  const ele = A[i];
  while (j >= 0 && A[j] > ele) {
    A[j + 1] = A[j];
    j--;
  }
  A[j + 1] = ele;
  console.log(A);
}
InsertSortRecur([7, 2, 3, 9, 5, 4], 5);

// 4.1-2 暴力求 maximum-subarray，O(n^^3)...
function bfMaximumSubarray(arr) {
  let maxSum = arr[0];
  const res = [0, 0, maxSum];
  const len = arr.length;
  for (let i = 0; i < len; i++) {
    for (let j = i; j < len; j++) {
      const curSum = Math.sum(arr.slice(i, j + 1));
      if (curSum > maxSum) {
        maxSum = curSum;
        res = [i, j + 1, maxSum];
      }
    }
  }
  return res;
}

// 4.1-5 sliding window 求 maximum-subarray, O(n)
// 以 DP 解法的思想为基础，maxsum_endat_i = Math.max(maxsum_endat_i-1 + arr[i], arr[i])
// 当遍历完所有 maxsum_endat_i，所有可能的 subArray 都已被考虑
function linearMaximumSubarray(arr) {
  const n = arr.length;
  let i = 0, j = 0, maxSum = 0, curSum = 0, res = [0, 0, 0];
  while (j < n) {
    curSum += arr[j];
    // 类似DP，直接在这里做出 maxSumEndAtHere 的抉择
    if (curSum < 0) {
      curSum = 0;
      i = j + 1;
    }
    if (curSum > maxSum) {
      maxSum = curSum;
      res = [i, j, maxSum];
    }
    j++;
  }
  return res;
}

// 6.2-2 最小堆化，针对一个不符合的 i 节点
function minHeapify(A, i) {
  const heapSize = A.heapSize; // 假定数组 A 有按照标准标记堆的大小
  const left = i * 2;
  const right = i * 2 + 1;
  if (left > heapSize) return;
  // 这里不直接使用 Math.min 是因为需要找出具体的 idx
  let minIdx = i;
  if (A[left] < A[minIdx]) minIdx = left;
  if (right <= heapSize && A[right] < A[minIdx]) minIdx = right;
  if (minIdx === i) return;
  [A[i], A[minIdx]] = [A[minIdx], A[i]]; // 思考题：为什么这里不用移位代替交换呢？因为不像 upward，有固定的 x.parent 目标比较。
  minHeapify(A, minIdx);
}

// 6.2-5 循环写法
// 如果觉得循环条件不好写，可以写 true，循环里面用 if、break、return 等表达终点
function minHeapifyLoop(A, i) {
  const heapSize = A.heapSize; // 假定数组 A 有按照标准标记堆的大小
  let cur = i, minIdx = i;
  while (true) {
    const left = cur * 2;
    const right = cur * 2 + 1;
    if (left > heapSize) return;
    if (A[left] < A[minIdx]) minIdx = left;
    if (right <= heapSize && A[right] < A[minIdx]) minIdx = right;
    if (minIdx === cur) return;
    [A[cur], A[minIdx]] = [A[minIdx], A[cur]];
    cur = minIdx;
  }
}

// 6.5-3 写一个教科书式的 Min-Heap 常规操作函数，包括 extract-min, heap-decrease-key, min-heap-insert
// 与大顶堆类似，一切都要从上述的 minHeapify 函数开始
// 注意按照堆的定义，A[0] 留空，A.heapSize 表示堆的大小，这里假定成员都是 number，而不是 reference，否则需要 deepCopy 给临时变量
// 从任意数组建堆
function buildMinHeap(A) {
  A.unshift(null);
  A.heapSize = A.length - 1;
  const midIdx = Math.floor(A.heapSize / 2);
  for (let i = midIdx; i >= 1; i--) {
    minHeapify(A, i);
  }
  return A;
}

// 取出堆顶最小元素
function extractMin(A) {
  const res = A[1];
  const heapSize = A.heapSize;
  A[1] = A[heapSize];
  minHeapify(A, 1);
  A.heapSize = A.heapSize - 1;
  return res;
}

// 将 i 节点优先级降低至 key 值
function heapDecreaseKey(A, i, key) {
  if (A[i] < A[key]) throw `node ${i} is already small than ${key}`;
  A[i] = key;
  // 不断向上尝试交换，找到自己的位置
  while (true) {
    const pIdx = Math.floor(i / 2);
    if (pIdx === 0) return;
    const parent = A[pIdx];
    if (parent > key) {
      [A[i], A[pIdx]] = [parent, key];
      i = pIdx;
    } else {
      return;
    }
  }
}

// 插入优先级为 key 的新成员
function minHeapInsert(A, key) {
  A.heapSize = A.heapSize + 1;
  A[heapSize] = Number.MAX_SAFE_INTEGER;
  heapDecreaseKey(A, heapSize, key);
}

// 6.5-6 优化 heapDecreaseKey，就像插入排序那样，将交换改写为移位，减少 2/3 赋值操作
function heapDecreaseKeyBetter(A, i, key) {
  if (A[i] < A[key]) throw `node ${i} is already small than ${key}`;
  A[i] = key;
  while (true) {
    const pIdx = Math.floor(i / 2);
    if (pIdx === 0) break;
    const parent = A[pIdx];
    // 只移位，不交换
    if (key < parent) {
      A[i] = parent;
      i = pIdx;
    } else {
      break; // 这里不能直接 return，因为最后留出的位置要赋值 key
    }
  }
  // 全部移位完成后，只需要最后赋值一次
  A[i] = key;
}

// 7-1 最早版本的分区算法 Hoare partition，它所构造的快排算法并不会将 分界点q 剔除在后续步骤里
// 以左侧为基准，从左到右找到大于等于基准的 i，从右到左找到小于等于基准的 j，交换它们。
// 它所返回的分界点并不是值为 pivot 的下标，每次都是不一定的，因此后面不能排除它自己
// 它所分成的两堆元素里，p...q 是小于等于 pivot，q+1...r 是大于 pivot
function hoarePartition(A, p, r) {
  const pivot = A[p];
  let i = p;
  let j = r;
  while (true) {
    while (A[i] < pivot) {
      i++;
    }
    while (A[j] > pivot) {
      j--;
    }
    if (i < j) {
      [A[i], A[j]] = [A[j], A[i]];
    } else {
      return j;
    }
  }
}

function quickSortUseHoare(A, p, r) {
  if (p < r) {
    const q = hoarePartition(A, p, r);
    quickSortUseHoare(A, p, q);
    quickSortUseHoare(A, q + 1, r);
  }
}

// 7-2 改写分区函数，返回两个分界点，从两堆改为三堆，左边小，中间等于，右边大于
// 其实就是经典的三色问题：单次遍历，小的交换去左边，大的交换去右边，额外标记右边的临时下标
// 需要注意的是：从右边交换回来的 j，需要再次比对，从左边交换回来的则不需要
function partitionFor3(A, p, r) {
  const pivot = A[r];
  let i = p, j = p, k = r;
  while (j <= k) {
    if (A[j] < pivot) {
      [A[i], A[j]] = [A[j], A[i]];
      i++;
    } else if (A[j] > pivot) {
      [A[j], A[k]] = [A[k], A[j]];
      k--;
      j--;
    }
    j++;
  }
  // 此时 i、k 就是右侧的分界点，i...k 是等于基准值的
  return [i, k];
}
// 改良后的快排，不再需要为相等的元素浪费时间
function quickSortFor3(A, p, r) {
  if (p < r) {
    const [q, k] = partitionFor3(A, p, r);
    quickSortFor3(A, p, q - 1);
    quickSortFor3(A, k + 1, r);
  }
}

// 8.x 写一个 counting sort，维护数组 B 记录值小于等于下标的元素数量
// 要求元素的大小范围都在 0...k
function countingSort(A, k) {
  const n = A.length;
  const B = Array(k).fill(null);
  // 构造 B 数组是有技巧的，借助前面位置的累积值
  for (let i = 0; i < n; i++) {
    if (A[i] === i) B[A[i]]++;
  }
  for (let i = 0; i < k - 1; i++) {
    B[i + 1] = B[i] + B[i + 1];
  }
  const res = Array(n).fill(null);
  // 倒序遍历 A 来填充 res
  for (let i = n - 1; i >= 0; i--) {
    const curCount = B[A[i]];
    res[curCount] = A[i];
    B[A[i]]--;
  }
  return res;
}

// 9.2-3 写一个迭代版的 Random-Select，原版是快排分区递归找 kth
// 注意快排自身并没有办法像这么简单改写为循环迭代写法
function randomSelectIterative(A, p, r, k) {
  while (true) {
    if (p === r) return A[p]; // 这个也是递归写法的终止条件之一
    const q = partitionFor3(A, p, r);
    if (k === q - p + 1) return A[q];
    if (k < q - p + 1) {
      r = q - 1;
    } else {
      p = q + 1;
      // 这个地方比较特殊，在有返回值的 random-select 写法里，k 在右侧递归是需要缩减的，因为它与 q-p+1 比较
      k = k - (q - p + 1);
    }
  }
}

// 10.2-7 经典的反转链表
function reverseList(L) {
  let cur = L.head, prev = null, next = null;
  while (cur) {
    next = cur.next;
    cur.next = prev;
    prev = cur;
    cur = next;
  }
}

function reverseListRecur(L, prev, cur, next) {
  if (!cur) return;
  next = cur.next;
  cur.next = prev;
  prev = cur;
  cur = next;
  reverseListRecur(L, prev, cur, next);
}

// 10.4-3 迭代遍历树，借助 stack
// 注意，前中后序有所不同，难度递增刚好是 pre-order < in-order < post-order
function preOrderIterative(T) {
  const stack = [T];
  while (stack.length) {
    const top = stack.pop();
    console.log(top.val);
    if (top.right)
      stack.push(top.right)
    if (top.left)
      stack.push(top.left)


  }
}
// 中序跟前序遍历完全不相干的思路
function inOrderIterative(T) {
  const stack = [];
  let cur = T;
  while (true) {
    // 先将左侧循环压入栈
    if (cur) {
      stack.push(cur);
      cur = cur.left;
    }
    else if (stack.length) {
      const top = stack.pop();
      console.log(top.val);
      cur = top.right;
    } else {
      break;
    }

  }

}

function postOrderIterative(T) {
  const stack = [];
  // TODO 
}

// 10.4-4 left-child, right-sibling 表示 N 叉树，遍历它
// 每个节点有四个指针，node.val, node.p, node.left, node.right-sibling
function traverseNtreePreOrder(T) {
  if (!T) return;
  console.log(T.val);
  traverseNtreePreOrder(T.left);
  traverseNtreePreOrder(T.right - sibling);
}

// 12.2-3 BST 找 predecessor，上一个比他小的节点
// 与 successor 类似，分两种情况：有合适 child 的，找其极值；否则向上找 parent。
function bstPredecessor(T, x) {
  if (x.left) return bstMax(x.left);
  let p = x.parent;
  while (p && p.left === x) {
    x = p;
    p = x.parent;
  }
  return p;
}

function bstMax(T) {
  if (!T.right) return T;
  return bstMax(T.right);
}

// 12.3-1 BST insert 递归写法
// 两个指针 cur 和 parent，向下寻找空位，cur 为 undefined 的时候就是空位
function bstInsertRecursive(T, x, cur, p) {
  if (!cur) {
    if (x.val < p.val) {
      p.left = x;
    } else {
      p.right = x;
    }
    x.p = p;
    return;
  }
  if (x.val < cur.val) {
    bstInsertRecursive(T, x, cur.left, cur);
  } else {
    bstInsertRecursive(T, x, cur.right, cur);
  }
}

// BST delete 的重要辅助函数：Transplant，用 y 替代 x 的位置。
// 注意这里只关注 parent，不处理 children，在 BST delete 中再处理。
function transplant(T, x, y) {
  if (!x.p) T.root = y;
  if (y) y.p = x.p;
  if (x.p.left === x) {
    x.p.left = y;
  } else {
    x.p.right = y;
  }
}

// 12-2 Radix tree 就是 Trie 字典树，只是路径上可能存在空节点，赋予特殊的 key
// 构造过程中如果左右有序，则 inorder 遍历可得有序集合


// 13.2-1 红黑树的辅助函数之一：右旋，一定有 left-child
function rightRotate(T, y) {
  const x = y.left;
  // 先交接子节点
  y.left = x.right;
  if (x.right) x.right.parent = y;
  // 再交接 parent 节点
  x.parent = y.parent;
  if (!y.parent) {
    T.root = x;
  } else if (y.parent.left === y) {
    y.parent.left = x;
  } else {
    y.parent.right = x;
  }
  // 最后交接自己
  x.right = y;
  y.parent = x;
}

// 13-1 Persistent-tree，一种可记忆之前状态的 BST，insert 时新建必经节点，复用不经过的节点，产生一颗新的树。
function persistentBSTreeInsert(T, k) {
  let newPrev = null, cur = T.root, newCur = new Node(cur.val), newTree = newCur;
  while (cur) {
    if (k < cur.val) {
      newCur.right = cur.right;
      cur = cur.left;
      const newNode = new Node(cur.val);
      newCur.left = newNode;
    } else {
      newCur.left = cur.left;
      cur = cur.right;
      const newNode = new Node(cur.val);
      newCur.right = newNode;
    }
    newPrev = newCur;
    newCur = newNode;
  }
  // cur 空节点，此时需要连接 newPrev 和 newCur
  if (!newPrev) return newTree;
  if (newCur.val < newPrev.val) {
    newPrev.left = newCur;
  } else {
    newPrev.right = newCur;
  }
  return newTree;
}

// 14.1-3 增强版红黑树每个节点有 size 属性，可以在 lgn 找出 kth 节点，比快排的 RandomSelect 快。
function OSselect(root, k) {
  let cur = root;
  let count = cur.left.size + 1;
  if (count === k) return cur;
  while (cur.left.size + 1 !== k) {
    if (cur.left.size + 1 > k) {
      cur = cur.left;
    } else {
      k = k - (cur.left + 1);
      cur = cur.right;
    }
  }
  return cur || null;
}

// 14.1-4 增强版红黑树 Order-Statistic Tree 求指定节点 k 的排位。
// 注意在红黑树的哨兵表示里，root.p = T.nil, leafNode.left = T.nil, leafNode.right = T.nil 
// 但是 T.nil 哨兵不会反向指向这些节点，也就是它没有 p、left、right 这些指针
function OSKeyRank(root, k) {
  function recurCountRank(root, k, count) {
    if (k.p.right !== k) return count;
    return recurCountRank(root, k.p, count + k.p.left.size + 1);
  }
  return recurCountRank(root, k, k.left.size + 1)
}

// 另一种增强版红黑树 Interval tree，每个节点维护一个区间，key 是区间下限，额外属性是 subTree 的区间上限 max
// 可以在 lgn 找出与区间 i 重叠的节点，且维护成本（插入、删除矫正）也是 lgn。
// 14.3-1 写出对应的左旋函数，维护 max 属性
function leftRotateForIntervalTree(T, x) {
  const y = x.right;
  x.right = y.left;
  if (x.right) x.right.p = x;
  y.p = x.p;
  if (x.p.left === x) {
    x.p.left = y;
  } else {
    x.p.right = y;
  }
  y.left = x;
  x.p = y;
  y.max = x.max;
  x.max = Math.max([...x.int.max, x.left.max, x.right.max]);
}

// 14.3-2 Interval-search 查找与指定区间 i 重叠的节点，要求区间是开放的，即不包括边界。
// 需要一个辅助函数 isOverlap 判断两区间是否重叠。max 属性帮助提前拒绝不符合条件的分支。
function intervalSearch(T, i) {
  let x = T.root;
  while (x && !isOverlap(x, i)) {
    if (x.left && x.left.max > i.min) {
      x = x.left;
    } else {
      x = x.right;
    }
  }
  return x;
}

function isOverlap(x, i) {
  if (!x || !i) return false;
  if (x.min > i.max || x.max < i.min) return false;
  return true;
}

// 14-2 约瑟夫排列，一个环状数组，循环取 mth，直到为空，组成新数组。
function JosephusPermutation(n, m) {
  const arr = Array.from({ length: n }, (v, i) => i + 1);
  arr.size = n;
  const res = [];
  let i = m - 1;
  while (arr.size) {
    i = checkArray(arr, i, res) + m;
  }
  return res;
}

// 辅助函数，输入下标，循环找数组元素，返回当前下标，更新数组
function checkArray(A, i, res) {
  const n = A.length;
  i = i % n;
  // 当前为空，顺位继续找
  while (A[i] === undefined) {
    i = (i + 1) % n;
  }
  // 找到后，更新数组属性，并返回当前下标
  A.size--;
  res.push(A[i]);
  A[i] = undefined;
  return i;
}

// 约瑟夫问题的教科书解法是 DP

// 15.1-3 切钢筋问题，P 数组表示每个长度能卖的价格，现在要求额外加入每一刀成本 c，求最大利润及其切法
// 需要注意的是 c 只能是常量，如果 c 是变量，则无法用 DP 求解，因为子问题相互影响。
// 加入数量限制 L 的话与 c 问题类似。
function curRod(n) {
  const memo = Array(n).fill(null);
  const path = Array(n).fill(null);
  const resVal = cutRodRecur(n, memo, path);
  const resPath = [];
  while (n > 0) {
    resPath.push(path[n])
    n = n - path[n]
  }
  return [resVal, resPath]
}

function cutRodRecur(n, memo, path) {
  if (memo[n]) return memo[n];
  if (n === 0) return 0;
  let curSum = -Infinity;
  for (let i = 0; i < n; i++) {
    // 这里判断切了一刀，但是 0 的时候没有切
    c = i === 0 ? 0 : c;
    if (P[i] + cutRod(n - i) - c > curSum) {
      curSum = P[i] + cutRod(n - i) - c;
      path[n] = i;
    }
  }
  memo[n] = curSum;
  return curSum;
}

// 15.3-6 货币兑换，将 1 通过一系列过程兑换为 n，如果手续费 c 为 0 或常量，则存在最优子结构
// 存在 k 使得 1...k 和 k...n 的兑换都是最优的
// 汇率 R 是一个二维数组，这里使用的是双边递归的思路。
function exchange(start, n, path, memo, c, originMoney, R) {
  if (start === n) return R[1][n] * originMoney;
  let cur = 0;
  for (let k = start; k <= n; k++) {
    if (k === start || k === n) {
      c = 0;
    } else c = c;
    const temp = exchange(start, k, path, c, originMoney, R) * exchange(k, n, path, c, originMoney, R) - c;
    if (temp > cur) {
      cur = temp;
      path[cur] = k;
    }
  }
  memo[start][n] = cur;
  return cur;
}
// 事实上可以单边递归解决，不需要 start 参数，还更简单一些，参考 cut-rod。

// 15.4-2 求 A、B 的最长公共子串 LCS，并找出路径。
// 根据条件不同，有不同的子问题需要处理。
// c[i,j] 表示 A[0...i] 与 B[0...j] 的 LCS 长度。
function findLCS(i, j, c, A, B) {
  if (c[i][j] > 0) return c[i][j];
  if (i === 0 || j === 0) return 0;
  let cur = 0;
  if (A[i] === B[j]) {
    cur = findLCS(i - 1, j - 1, c, A, B) + 1;
  } else {
    // 如果需要维护具体的路径缓存，则不能使用 max
    cur = Math.max(findLCS(i - 1, j, c, A, B), findLCS(i, j - 1, c, A, B));
  }
  c[i][j] = cur;
  return cur;
}

function findLCSPath(c, A, B) {
  const path = [];
  let i = A.length, j = B.length;
  while (i > 0 && j > 0) {
    if (c[i][j] === c[i - 1][j - 1] + 1) {
      path.unshift(A[i]);
      i--;
      j--;
    } else if (c[i][j] === c[i - 1][j]) {
      i--;
    } else {
      j--;
    }
  }
  return path;
}

// 15.4-4 如果只求 LCS 长度，不需要具体 path，c 能否使用一维数组？
// 答案是不能，但我们可以清空 i-2 及 j-2 之前的位置，因为用不到了。

// 15.4-6 求最长递增子串，当我们说子串或 subsequence 的时候，默认不要求连续。
// 其中 c[i] 表示以 i 结尾的最长递增子串长度，它的最优子结构并不好找，并不是看 i 与 i-1 的关系，而是
// c[i] = 1 + max(c[j])，其中 j < i 且 A[j] < A[i]
// 最后遍历 c 找到最大的值 max，然后从下标 max 开始倒序遍历 path 即可获取具体的子串。
function monoIncrSub(A, i, memo, path) {
  if (memo[i]) return memo[i];
  if (i === 0) return 1;
  let cur = 0;
  for (let j = 0; j < i; j++) {
    if (A[j] < A[i]) {
      const cj = monoIncrSub(A, j, memo, path);
      if (cj > cur) {
        cur = cj;
        path[cur] = A[j];
      }
    }
  }
  memo[i] = cur + 1;
  return memo[i]
}

// 15-5 编辑距离，是判断字符串相似的另一种方式（第一种是上述的 LCS）
// 从左开始，A[i] !== B[j] 有几种处理方式：
// 跳过，即 i++
// 替换，即 Z[j] = A[i] 然后 i++, j++
// 如果 A[i]、A[i+1] 与 B[j+1]、B[j] 相等，尝试翻转，即 Z[j] = A[i+1], Z[j+1] = A[i], 然后 i+2, j+2
// 这里假定相等和替换的成本是 1，其他操作的成本是 2，我们从左开始递归，找出最小成本
function editDistance(i, j, memo, Z, A, B) {
  if (memo[i][j]) return memo[i][j];
  if (i >= A.length || j >= B.length) return 0;
  let curDistance = Infinity;
  if (A[i] === B[j]) {
    curDistance = editDistance(i + 1, j + 1, memo, Z, A, B) + 1;
  } else {
    const replace = editDistance(i + 1, j + 1, memo, Z, A, B) + 1;
    const skip = editDistance(i + 1, j, memo, Z, A, B) + 2;
    if (replace < curDistance) {
      curDistance = replace;
      Z[i] = 'replace';
    }
    if (skip < curDistance) {
      curDistance = skip;
      Z[i] = 'skip';
    }
    if (A[i] === B[j + 1] && A[i + 1] === B[j]) {
      const twiddle = editDistance(i + 2, j + 2, memo, Z, A, B) + 2;
      if (twiddle < curDistance) {
        curDistance = twiddle;
        Z[i] = 'twiddle';
      }
    }
  }
  memo[i][j] = curDistance;
  return curDistance;
}

// 15-6 party 邀请，要求不能同时邀请 i 及其直属下属，求最大的快乐值。
// 每个节点有 val 和 sub 属性，表示快乐值和下属
// 对每个节点，我们有两种选择，邀请他或者不邀请他，最值的计算方式即最优子结构：
// c[i] = max(sumOf c[j], i.val + sumOf d[j])，其中 j 是 i 的下属，d[j] 表示不邀请 j
// d[i] = sumOf c[j]，这里可以观察到 c[i] 最终还是可以由它的下属们 j、k 来推导，而 d 只是一个临时概念
// 这里 planParty 计算的就是 c[i]
function planParty(i, memo, path) {
  if (memo[i]) return memo[i];
  if (!i) return 0;
  let curSum = 0;
  // 不邀请 i 本人
  for (let j = 0; j < i.sub.length; j++) {
    curSum += planParty(i.sub[j], memo, path);
  }
  // 邀请本人
  let d = i.val;
  for (let j = 0; j < i.sub.length; j++) {
    // 直属下属 j 都不能被邀请，因此只能考虑 j 的下级
    for (let k = 0; k < j.sub.length; k++) {
      d += planParty(j.sub[k], memo, path);
    }
  }
  // 最值推导
  if (d > curSum) {
    curSum = d;
    path[i] = true;
  }
  memo[i] = curSum;
  return curSum;
}

// 15-8 图片压缩，m * n 像素，要求相邻 row 删除的点必须是直线或对角线，求最小损失的路径。
// 每个点有 val 表示被删除后的损失。
// 子结构定义：c[i,j] 表示从最开始到这个点的路径的最值，最终从 c[n,i] 找出最小值就是答案。
// 最优子结构：c[i,j] = min(c[i-1,j-1], c[i-1,j], c[i-1,j+1]) + val[i,j]
// 越界哨兵，将越界的点 val 设置为 infinity
function imageCompress(i, j, memo) {
  if (memo[i][j]) return memo[i][j];
  if (i < 0 || i > m || j < 0 || j > n) return MAX_SAFE_INTEGER;
  memo[i][j] = val[i][j] + Math.min(...[imageCompress(i - 1, j - 1), imageCompress(i - 1, j), imageCompress(i - 1, j + 1)]);
  return memo[i][j];
}

// 15-9 切割字符串，每次的成本是当前的长度，总长 n 的字符串，指定下标 1...m 为切割点，求最小成本和切割顺序。
// 能否用 DP 解决？最值问题，有 m! 种切法，相邻有联系，有重复子问题
// 最优子结构：c[i,j] 表示 i block 到 j block 的最小成本，其中 j <= m+1，也就是说 m 提前将字符串分成了 m+1 块
// 最值的推导关系：c[i,j] = min(c[i,k] + c[k+1,j] + len[i,j]) for i <= k <= j


// 16.2-2 经典的 0-1 背包问题，每个物品重 weight[i] 价值 val[i]，要求不超过总重量 W 的最大价值
// 最优子结构：c[i,j] 表示选到第 i 个物品时，重量不超过 j 的最大价值
// 最值的推导：c[i,j] = max(c[i-1,j-wei[i]]+val[i], c[i-1,j])，即可能拿或者不拿物品 i，取两种选择里较大的一种
function zeroOneBag(i, j, memo, path, wei, val) {
  if (i === 0 || j <= 0) return 0;
  if (memo[i][j]) return memo[i][j];
  let cur = 0;
  let pickMe = zeroOneBag(i - 1, j - wei[i], memo, path, wei, val) + val[i];
  let notPickMe = zeroOneBag(i - 1, j, memo, path, wei, val);
  if (pickMe > notPickMe) {
    path[i] = true;
    cur = pickMe;
  } else {
    path[i] = false;
    cur = notPickMe;
  }
  memo[i][j] = cur;
  return cur;
}

// 16.2-6 碎片背包问题，适用贪婪算法
// 按单价从高到低排序，对于当前物品 i，有两种可能：
// 加上 wei[i] 后没有超重，则拿光 i
// 超重，则拿到满重量
function fractionalBag(i, j, res, wei, val) {
  if (i > n) return res;
  if (j + wei[i] < W) {
    return fractionalBag(i + 1, j + wei[i], res + val[i], wei, val);
  } else {
    return res + (W - j) * val[i] / wei[i];
  }
}


// 第21章，不相交集合的实现，Disjoint-set forests
// 注意 root 的特点是 x.parent = x，通常是其所在集合的代表



function makeSet(x) {
  x.parent = x;
  x.rank = 0;
  return x;
}

function unionSets(u, v) {
  return linkSets(findSet(u), findSet(v));
}

// 这个就是 union by rank，防止树太高
function linkSets(u, v) {
  if (u.rank < v.rank) {
    u.parent = v;
    v.rank = Math.max(v.rank, u.rank + 1); // 这行是多余的，因为 u.rank + 1 <= v.rank
    return v;
  } else {
    v.parent = u;
    if (u.rank === v.rank) u.rank++; // 有趣的是只有二者 rank 相等且合并时，rank 才有增加
    return u;
  }
}

// 这个就是 path-compression，路径压缩，加快下次查找
function findSet(x) {
  if (x === x.parent) return x;
  x.parent = findSet(x.parent);
  return x.parent;
}


// 21-1 Offline minimum 问题，静态数据源，给定一系列操作由 n 次 Insert 和 m 次 Extract-min 组成，数值在 1..n 之间，求 extracted 数组。
// S 的组成比如：2,3,1,E,5,E,E,7,8,E 等等
// 因为是静态数据，我们提前知道所有的行为和数据，可以进行预处理，将连续的 Insert 抽象成一个集合 j
function offlineMinimum(S, n, m) {
  function treeNode(v) {
    return { v }
  }
  const initNodes = Array(n + 1).fill(null);
  // 先把数字构造成 node，存起来，防止重复创建以及引用丢失
  for (let i = 1; i <= n; i++) {
    const node = treeNode(i)
    initNodes[i] = node;
  }
  const kSets = [];
  let countE = 0;
  // 遍历 S 把连续的 Insert 合并为一个集合
  for (let i = 0; i < S.length; i++) {
    const ele = S[i];
    if (ele !== 'E') {
      const eleSet = makeSet(initNodes[ele]);
      if (!kSets[countE]) {
        kSets[countE] = eleSet;
      } else {
        kSets[countE] = linkSets(kSets[countE], eleSet)
      }
      // 给集合挂载额外下标，最后用于生成 extracted 数组
      kSets[countE].countE = countE;
    } else {
      countE++;
    }
  }
  // 从小到大，为数字 i 找到它所在的 extracted 数组的位置，只有小的有位置
  // 注意这期间还需要不断的合并集合，因为它的编号透露了 E 的次数
  const extracted = [];
  for (let i = 1; i <= n; i++) {
    const node = initNodes[i];
    const mySet = findSet(node);
    extracted[mySet.countE] = i;
    // TODO 尝试找到下一个集合，让它吞并当前集合，为下一个 E 做准备，因为当前的 countE 已经不能用了
  }

}


// 一个重要的问题：如何用循环 + stack 表示递归，体现为代码？
// 在函数调用栈里，假设函数 A 执行到一半，遇到另一个函数 B 调用，此时 A 会暂停，将 B 的内容构造为一个对象压入调用栈
// 注意不是所有函数一次性全部压入栈中，而是边执行边压栈，只有遇到新的函数调用才会压栈
// 在 DFS 的 stack 写法中，并不是直接翻译递归代码的函数调用栈过程，而是重新组织一种逻辑，破坏了前序遍历的顺序，好处是代码比较好写

// 在图的 DFS/BFS 教科书写法里，用白、灰、黑三种颜色来标记节点状态，防止重复，且为其他应用铺垫，一开始都是白色
// BFS 只会从一个指定顶点开始，寻找到达目标的最短路径，而 DFS 通常遍历所有白色的顶点开始，且 DFS 使用全局变量 time 来标记时间 

function DFSDefault(G) {
  time = 0;
  for (const u of G.V) {
    if (u.color === 'white') {
      DFSRecur(G, u)
    }
  }
}

function DFSRecur(G, u) {
  time++;
  u.start = time;
  u.color = 'gray';
  for (const v of G.adj[u]) {
    if (v.color = 'white') {
      v.prev = u;
      DFSRecur(G, v);
    }
  }
  time++;
  u.finish = time;
  u.color = 'black';
}

function DFSVisitStack(G, u) {
  const stack = [];
  stack.push(u);
  time++;
  u.start = time;
  u.color = 'gray';
  while (stack.length > 0) {
    const top = stack.pop();
    const v = whiteNeighbor(G, top);
    if (!v) {
      top.color = 'black';
      time++;
      top.finish = time;
    } else {
      v.color = 'gray';
      v.prev = top;
      time++;
      v.d = time;
      stack.push(v);
    }
  }
}

// 为什么说二叉树的 DFS 遍历复杂度是 O(n) ？这要从两方面分析，一是每个节点需要入栈和出栈一次，贡献了 2n 复杂度
// 二是每个节点有两条边，就是通常说的每个节点被访问两次，也贡献了 2n 复杂度，因此严格来说是 4n 复杂度
// 如果是 N 叉树，每个节点有 N 条边，则每个节点被访问 N 次，根据图的 O(V+E) 类比，其 DFS 复杂度是 O(n+n*N)
// 简而言之，图的 DFS、BFS 算法具有更广阔的适用性，树只是它的简化版


// 22.3-10 改造 DFS 算法，对 edges 进行分类，有向图的 edges 可以分为四类：tree、back、forward、cross edges
// 依据是第一次遇到 v 的时候，检查 v 的颜色：白色是 tree，灰色是 back，黑色是其他两种，有趣的是如果 v.start > u.start，说明是 forward edge
function DFSClassifyEdges(G, u) {
  u.color = 'gray';
  time++;
  u.start = time;
  for (const v of G.adj[u]) {
    if (v.color === 'white') {
      treeEdges.push([u, v]);
      v.prev = u;
      DFSClassifyEdges(G, v);
    } else if (v.color === 'gray') {
      backEdges.push([u, v]);
    } else if (v.start > u.start) {
      forwardEdges.push([u, v]);
    } else {
      crossEdges.push([u, v])
    }
  }
  time++;
  u.finish = time;
  u.color = 'black';
}

// 22.3-12 改造 DFS 把无向图分成 Disjoint-Sets，每个节点挂载 .cc 属性，表示所处的集合代号
function DFSToDisjointSets(G) {
  let k = 0;
  for (const u of G.V) {
    if (u.color === 'white') {
      k++;
      recur(G, u, k);
    }
  }

  function recur(G, u, k) {
    u.color = 'gray';
    u.cc = k;
    for (const v of G.adj[u]) {
      if (v.color === 'white') {
        v.prev = u;
        recur(G, v, k);
      }
    }
    u.color = 'black';
  }
}

// 22.4-2 改造 DFS 计算有向无环图 dag 中，从 p 到 v 有几条路
function DFSCountPToV(G, p, v) {
  let count = 0;
  function recur(G, p, v) {
    p.color = 'gray';
    time++;
    p.start = time;
    for (const u of G.adj[p]) {
      if (u === v) {
        count++;
      } else if (u.color === 'white') {
        u.prev = p;
        recur(G, u, v);
      }
    }
    p.color = 'black';
    time++;
    p.finish = time;
  }

  recur(G, p, v);
  return count;
}

// 22.4-5 另一种拓扑排序的实现思路：每次找到 in-degree 为 0 的节点，说明它不依赖别人，将它从图中移除，并将它指向别人的边切断，它就是 res 序列的早期成员。
function TopoByIndegree(G) {

}