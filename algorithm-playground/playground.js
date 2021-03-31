// 两个二进制数相加
// 其实就是链表相加的思路，从尾巴开始逐位相加，标记是否进位
function sumTwoBinary(arr1, arr2) {
  const m = arr1.length;
  const n = arr2.length;
  // 标记是否进位
  let plus = 0;
  let i = m - 1;
  let j = n - 1;
  let res = [];
  while (i >= 0 || j >= 0) {
    // 使用哨兵避免后面剩余数组的判断
    // TODO: 更好的哨兵做法是，提前将较短的数组头部补齐0，就不用每次循环里判断
    let v1 = i >= 0 ? arr1[i] : 0,
      v2 = j >= 0 ? arr2[j] : 0;
    let curSum = v1 + v2 + plus;
    let curNum = curSum % 2;
    plus = Math.floor(curSum / 2);
    res.unshift(curNum);
    i--;
    j--;
  }
  return res;
}

// sumTwoBinary([1, 0, 1, 1, 0], [0, 1, 0]);

// 插入排序的递归写法
// 先排序 f(n-1), 然后将 A[n] 插入
function insertionSortRecursive(arr) {
  const n = arr.length;
  if (n === 1) return arr;
  // TODO: 可以用更多参数，来避免所占内存空间的伸缩，也就是原地排序。
  // 但是不要使用 arr 作为载体，会有 reference 被修改导致的 bug。
  const subArr = insertionSortRecursive(arr.slice(0, n - 1));
  const last = arr[n - 1];
  subArr.push(last);
  let index;
  for (index = n - 1; index > 0; index--) {
    if (subArr[index - 1] > last) {
      subArr[index] = subArr[index - 1];
    } else break;
  }
  subArr[index] = last;
  return subArr;
}

// console.log(insertionSortRecursive([3, 1, 2, 5, 4]));

// 二分查找的递归写法
// 判断中间值是不是目标，否则查找左边和右边
function binarySearchRecursive(arr, start, end, target) {
  if (start > end) return -1;
  const mid = start + Math.floor((end - start) / 2);
  if (arr[mid] === target) return mid;
  // tricky part: 因为不存在返回 -1，我们尽量返回存在的结果
  return Math.max(
    binarySearchRecursive(arr, start, mid - 1, target),
    binarySearchRecursive(arr, mid + 1, end, target)
  );
}

// console.log(binarySearchRecursive([1, 2, 3, 4, 10], 0, 4, 4));

// 二分查找变种，递归写法
// 找到第一个大于 target 的元素位置
function binarySearchVariantRecursive(arr, start, end, target) {
  if (start > end) return -1;
  const mid = start + Math.floor((end - start) / 2);
  if (arr[mid] > target) {
    // 防止越界
    if (mid === 0) return 0;
    if (arr[mid - 1] <= target) return mid;
    return binarySearchVariantRecursive(arr, start, mid - 1, target);
  }
  return binarySearchVariantRecursive(arr, mid + 1, end, target);
}

// console.log(binarySearchVariantRecursive([1, 2, 3, 5, 10], 0, 4, 4));

// 插入排序，二分查找优化版本
// 在已排序的数组中使用二分查找定位，而不是遍历移位
function insertionSortWithBinarySearch(arr) {
  const n = arr.length;
  if (n === 1) return arr;
  const subArr = insertionSortWithBinarySearch(arr.slice(0, n - 1));
  const last = arr[n - 1];
  subArr.push(last);
  // 使用变种二分查找，正序找到最后一个小于或者第一个大于 last 的位置
  const to = binarySearchVariantRecursive(subArr, 0, n - 2, last);
  // 如果返回 -1 说明位置不用换
  if (to === -1) return subArr;
  for (let index = n - 1; index > to; index--) {
    subArr[index] = subArr[index - 1];
  }
  subArr[to] = last;
  return subArr;
}

// console.log(insertionSortWithBinarySearch([3, 1, 7, 5, 10], 5));

// 求给定数组中的逆序对数量，逆序对定义：i < j 且 a[i] > a[j]
// 等同于插入排序中移位的次数 n*n，使用变种二分查找辅助的话 nlgn
function inversions(arr) {
  let res = 0;
  // 常规插入排序
  function recur(arr) {
    const n = arr.length;
    if (n === 1) return arr;
    const subArr = recur(arr.slice(0, n - 1));
    const last = arr[n - 1];
    subArr.push(last);
    let i = n - 1;
    while (i > 0) {
      if (subArr[i - 1] > last) {
        subArr[i] = subArr[i - 1];
        res++; // 移位次数
        i--;
      } else break;
    }
    subArr[i] = last;
    return subArr;
  }
  recur(arr);
  return res;
}

// console.log(inversions([6, 5, 4, 3, 2, 1]));

// 解法二：使用 mergeSort 的 merge 函数
// arr1 和 arr2 虽然各自排序，但元素相对另一个集合的位置是没变的
// 因此可以利用 arr2 的元素在 merge 过程中的优先次数，来计算逆序对
// 总的来说，两边先求各自的逆序对，并不影响继续求总的逆序对
function inversionsByMerge(arr) {
  // 改造 merge
  function merge(arr1, arr2) {
    const m = arr1.length,
      n = arr2.length;
    let i = 0,
      j = 0,
      sorted = [];
    while (i < m || j < n) {
      const v1 = i < m ? arr1[i] : Number.MAX_SAFE_INTEGER,
        v2 = j < n ? arr2[j] : Number.MAX_SAFE_INTEGER;
      if (v1 > v2) {
        sorted.push(v2);
        // 只有当 arr1 还有剩余才构成逆序，注意此刻新增逆序对数 m-i，因为 i 之后的元素都与 j 构成逆序对
        if (v1 !== Number.MAX_SAFE_INTEGER) res += m - i;
        j++;
      } else {
        sorted.push(v1);
        i++;
      }
    }
    return sorted;
  }
  function mergeSort(arr, start, end) {
    if (start > end) return [];
    if (start === end) return [arr[start]];
    const mid = start + Math.floor((end - start) / 2);
    const left = mergeSort(arr, start, mid);
    const right = mergeSort(arr, mid + 1, end);
    return merge(left, right);
  }
  let res = 0;
  mergeSort(arr, 0, arr.length - 1);
  return res;
}

// console.log(inversionsByMerge([6, 5, 4, 3, 2, 1]));

// 分治思想的一个应用
// maximum subarray 问题拆分三个子问题：
// 要么在左半边，要么在右半边，要么在中间经过 mid 点，找出三者中最大的即可
function maximumSubarray(arr, start, end) {
  // TODO
}

// maximum subarray 的 sliding window 解法：
// i, j 指针构造的区间，如果 sum <=0 全部放弃，从零开始!
// 总是尝试维持一个 maximum
function maximumSubarraySlidingWindow(arr) {
  let i = 0,
    j = 1,
    low = 0,
    high = 0,
    n = arr.length,
    maxSum = arr[0],
    curSum = arr[0];
  while (j < n) {
    if (arr[j] + curSum > 0) {
      curSum += arr[j];
      if (curSum > maxSum) {
        maxSum = curSum;
        low = i;
        high = j;
      }
      j++;
    } else {
      // 亮点在这里，放弃一切
      curSum = 0;
      j++;
      i = j;
    }
  }
  return [low, high, maxSum];
}

// 手写 min heap，支持：heapifyDown、build、extract、decreasePriority、insert, heapSort
// 注意，如果希望空间复杂度 O(1)，那么需要在 arr 里单独标记真实的堆大小 heapSize
class minHeap {
  constructor(arr) {
    // 首位留空，方便父子节点的下标计算
    arr.unshift(null);
    // 表示真实的 heap 大小
    this.heapSize = arr.length - 1;
    this.heap = this.build(arr);
  }
  build(arr) {
    const mid = Math.floor(arr.length / 2);
    for (let index = mid; index > 0; index--) {
      this.heapifyDown(arr, index);
    }
    return arr;
  }
  // 向下堆化，不断与更小的子节点交换
  heapifyDown(arr, index) {
    while (2 * index <= this.heapSize) {
      // 要小心 2*index + 1 越界的情况
      const smallerChild = Math.min(
        arr[2 * index],
        arr[2 * index + 1] || Number.MAX_SAFE_INTEGER
      );
      if (arr[index] > smallerChild) {
        const childIdx =
          smallerChild === arr[2 * index] ? 2 * index : 2 * index + 1;
        [arr[index], arr[childIdx]] = [arr[childIdx], arr[index]];
        index = childIdx;
      } else break;
    }
  }
  // 删除 root，将尾巴放到 root 位置，然后向下堆化
  extract(heap) {}
  // 向上堆化，通常定义为改变优先级的函数
  decreasePriority(heap, index, value) {}
  // insert 新元素，先 push，然后向上堆化
  insert(heap, value) {}
  // 堆排序，原地排序 nlogn
  // TODO?
  heapSort() {
    while (this.heapSize > 1) {
      // 循环执行：头尾交换，堆缩小，堆化
      [this.heap[1], this.heap[this.heapSize]] = [
        this.heap[this.heapSize],
        this.heap[1]
      ];
      this.heapSize--;
      this.heapifyDown(this.heap, 1);
    }
    this.heap.shift();
    return this.heap;
  }
}

// const heap = new minHeap([3, 1, 5, 2, 8, 4]);
// console.log(heap.heap);
// console.log(heap.heapSort());

// Young tableaus 是一种特殊的二维矩阵，横竖两个方向都是递增的，空的格子用 Infinity 占据
// 参照 heap，写出 extract、insert 等方法
// TODO 查找的话利用二分查找及其变种
class YoungTableaus {
  constructor(arr) {}
}

// 快排分区函数变种，要求标记相同元素的区间
// 类似三色划分，新增了右边的指针，标记较大的数字
function partitionThree(arr, p, r) {
  // i 正向表示小的，j 逆向标记大的，k 是主指针
  let i = p,
    j = r,
    pivot = arr[r];
  for (let k = p; k <= j; k++) {
    const element = array[k];
    if (element < pivot) {
      [arr[k], arr[i]] = [arr[i], arr[k]];
      i++;
    } else if (element > pivot) {
      [arr[k], arr[j]] = [arr[j], arr[k]];
      j--;
      k--; // k 在下回合需要重新被检查
    }
  }
  return [i, j];
}

// 随机分区函数
function randomPartition(arr, p, r) {
  const ran = p + Math.round((r - p) * Math.random());
  [arr[ran], arr[r]] = [arr[r], arr[ran]];
  partition(arr, p, r);
}

// 常规分区函数
function partition(arr, p, r) {
  let i = p;
  // etc
}

// counting sort，适合上下限范围不大的数据排序
// 经典的将 val 作为 index 构造数组
function countingSort(arr) {
  const top = Math.max(...arr);
  const temp = Array(top + 1).fill(0); // 存储 val <= index 的元素个数(对 arr 而言)
  for (const v of arr) {
    // 初始化，记录 index = val 的个数(对 arr 而言)
    temp[v] += 1;
  }
  for (let i = 0; i < top; i++) {
    // 累加计算出 val <= index 的个数(对 arr 而言)
    temp[i + 1] += temp[i];
  }
  const res = Array(arr.length).fill(0);
  for (const val of arr) {
    // 遍历原数组，因为有 temp[val] 个元素 <= val, 因此有序的 res 该位置一定是 val
    res[temp[val]] = val;
    temp[val]--;
  }
  return res;
}
// console.log(countingSort([3,0,1,9,10,2,7]));

// N branch tree 的一种表示方式
class Node {
  constructor(val) {
    this.val = val;
    this.left = null;
    this.rightSibling = null; // 指向同级
  }
}

function traverseN(root, res) {
  if (!root) return;
  res.push(root.val);
  traverseN(root.left, res);
  traverseN(root.rightSibling, res);
}

// BST insert
function bstInsert(root, node) {
  let cur = root;
  let p = null;
  while (cur) {
    const val = cur.val;
    p = cur;
    if (node.val <= val) {
      cur = cur.left;
    } else {
      cur = cur.right;
    }
  }
  // 经过比较，cur 一定会走到边界点，而 p 一定至多只有一个子节点
  if (!p) return node;
  if (node.val <= p.val) {
    p.left = node;
  } else {
    p.right = node;
  }
  return root;
}

// recusive
function bstInsertRecur(root, node) {
  function recur(cur, p, root, node) {
    // 结束条件
    if (!cur) {
      if (!p) return node;
      if (node.val <= p.val) {
        p.left = node;
      } else {
        p.right = node;
      }
      return root;
    }
    // 子问题，答案就在其中一种，不需要组合
    if (node.val <= cur.val) {
      return recur(cur.left, cur, root, node);
    } else {
      return recur(cur.right, cur, root, node);
    }
  }
  return recur(root, null, root, node);
}

// 二分查找递归写法
function bsR(arr, low, high, target) {
  if (low > high) return -1;
  const mid = low + Math.floor(high - low / 2);
  if (arr[mid] === target) return mid;
  if (arr[mid] < target) {
    return bsR(arr, mid + 1, high, target);
  } else {
    return bsR(arr, low, mid - 1, target);
  }
}

// N叉树的 right-sibling 表示
class NBranchTreeNode {
  constructor(val) {
    this.val = val;
    this.parent = null;
    this.left = null;
    this.rightSibling = null;
  }
}
// 遍历它
function traverseIt(root) {
  if (!root) return;
  console.log(root.val);
  traverseIt(root.left);
  traverseIt(root.rightSibling);
}
