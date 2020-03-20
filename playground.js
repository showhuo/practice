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
    let curNum = curSum % 2,
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
  // TODO: 可以用更多参数，来避免所占内存空间的伸缩，也就是原地排序。不要使用 arr 作为载体，要小心引用被修改的问题。
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

// i < j 且 a[i] > a[j]，称为一个 inversion 逆序对
// 求给定数组中的逆序对数量
// 等同于插入排序中移位的次数 n^^2，使用变种二分查找辅助的话 nlgn
// 也可以在归并排序的 merge 函数中解决 nlgn
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

// 使用 mergeSort 的 merge 解决
// arr1 和 arr2 虽然各自排序，但元素相对另一个集合的位置是没变的
// 因此可以利用 arr2 的元素在 merge 过程中的优先次数，来计算逆序对
// 换一种说法就是，两边先求各自的逆序对，并不影响继续求总的逆序对
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

// 分治思想的几个应用
// 股票买卖，只能操作一次……
// 股票有涨有跌，我们希望找到涨幅最好的区间
// 需要转换为 maximum subarray 问题：求连续子序列使得 sum 最大……
// 三个子问题：要么在左半边，要么在右半边，要么在中间经过 mid 点，找出三者中最大的即可
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

// 手写一个 min heap，支持：heapifyDown、build、extract、decreasePriority、insert, heapSort
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

const heap = new minHeap([3, 1, 5, 2, 8, 4]);
console.log(heap.heap);
console.log(heap.heapSort());

// Young tableaus 是一种特殊的二维矩阵，横竖两个方向都是递增的，空的格子用 Infinity 占据
// 参照 heap，写出 extract、insert 等方法
// 如何判断 target 是否在二维矩阵里？要求复杂度 m+n
// TODO
class YoungTableaus {
  constructor(arr) {}
}
