// 给定一个链表，每 k 个节点一组进行翻转，如果节点总数不是 k 的整数倍，最后的片段不需要翻转
// 考察链表和递归
// 难度：hard
var reverseKGroup = function(head, k) {
  if (!head) return null;
  let prev = null;
  let cur = head,
    i = 0;
  while (cur && i < k) {
    const next = cur.next;
    cur.next = prev;
    prev = cur;
    cur = next;
    i += 1;
  }
  // !剩余的片段不够 k 个，把该片段再翻转回来
  if (i !== k) {
    while (prev) {
      const next = prev.next;
      prev.next = cur;
      cur = prev;
      prev = next;
    }
    return cur;
  }
  head.next = reverseKGroup(cur, k);
  return prev;
};

// 给定一组数字，和一个窗口大小 k，求窗口在滑动的过程中，每次的最大值组成的数组
// 考察 queue、monoqueue
// 难度：hard
var maxSlidingWindow = function(nums, k) {
  // 总共滑动 n-k+1 次，相邻的数字 i 和 i+1，如果 nums[i] < nums[i+1]，那么 i 永远没机会当最大值
  // 因此 queue[-1] 与后来的值比较，遇到大值则 queue.pop()，只保留有机会成为最大值的元素
  // queue[0] 总是最大的，但是会过期，此时需要 queue.shift()
  const n = nums.length,
    queue = [];
  if (!n || !k) return [];
  const res = Array(n - k + 1).fill(null);
  // !考察 i 的时候，滑动窗口的起点分别对应 i-k+1
  for (let i = 0; i < n; i++) {
    // 检查过期的 head
    while (queue && queue[0] < i - k + 1) {
      queue.shift();
    }
    // 拿出没希望成为最大值的
    while (queue && nums[queue[queue.length - 1]] < nums[i]) {
      queue.pop();
    }
    // 潜在值都放入队列
    queue.push(i);
    // 开始取值
    if (i >= k - 1) res[i - k + 1] = nums[queue[0]];
  }
  return res;
};

// 接雨水，每个水槽宽度 1
var trap = function(height) {
  // 虽说经典解法是 monoStack，但显然不如最优的双指针解法容易理解
  // left、right 指针互相逼近，维护额外的 leftMax 和 rightMax 指针，总共四个指针
  // !当 rightMax > leftMax，说明只有左边的雨水可以蓄住（较矮），此时从左边往右累计水槽
  // 当 left < leftMax，累加一次水槽 leftMax-left
  // 否则说明没有水槽，更新 leftMax = left，此时与 rightMax 比较，再确认从哪边开始累计
  // 重复上述过程，直到 left、right 相遇
  let left = 0,
    right = height.length - 1,
    leftMax = height[0],
    rightMax = height[right];
  let res = 0;
  while (left <= right) {
    if (rightMax > leftMax) {
      if (height[left] < leftMax) {
        // 左边蓄水一次
        res += leftMax - height[left];
      } else {
        leftMax = height[left];
      }
      left++;
    } else {
      if (height[right] < rightMax) {
        // 右边蓄水一次
        res += rightMax - height[right];
      } else {
        rightMax = height[right];
      }
      right--;
    }
  }
  return res;
};
