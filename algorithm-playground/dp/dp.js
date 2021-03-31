// 动态规划
// 是求解多阶段决策、最优解问题的三种思想之一（贪心、回溯）
// 要求每阶段的状态有重复
// 如果每次决策可选项不超过 2，可以考虑循环填充二维数组解决（提前分析出状态规模上限）
// 反之请找出状态转移方程

// 0-1 背包升级版，给出两个数组分别表示物品的重量和价值
// 在不超过最大限重的前提下，找到最大价值组合
function getMaxV(weights, values, maxW) {
  // 思路：每次决策是拿或者不拿当前物品，选项只有两种，可以使用状态转移表
  // 用 row 表示第几步决策，用 column 表示物品的价值，可根据价值上限(求和)给出数组 size
  // !每个单元格我们存储一个二元数组，表示当前组合的价值和重量
  // 最后从价值最高的尾巴倒序寻找，重量符合要求的第一个格子
  // !但是，我们应该尽可能精确上限，所以使用题目限制条件，即 maxW 作为上限更合适
  // 也就是说，应该用 column 表示物品的重量，在满足重量的情况下，找出最大价值
  const n = weights.length;
  const matrix = Array(n).fill(Array(maxW + 1).fill(null));
  matrix[0][0] = [0, 0]; // 第一格意思是第一件物品不拿
  if (weights[0] <= maxW) {
    matrix[0][weights[0]] = [weights[0], values[0]]; // 意思是第一件拿下
  }
  // 因为需要依赖上一个选择，所以只能从第二个物品开始，循环填充
  for (let i = 1; i < n; i++) {
    // 做出选择：不拿当前物品，因此维持上一步的状态
    // 因为这一步先执行，占据的格子有可能被下面的选择覆盖，这就是触发了去重的行为
    for (let j = 0; j < maxW + 1; j++) {
      if (matrix[i - 1][j]) matrix[i][j] = matrix[i - 1][j];
    }
    // 做出选择：拿下当前物品，基于上一步的状态计算获取当前状态
    // 这里的 j 还要注意不能越界
    for (let j = 0; j < maxW + 1 - weights[i]; j++) {
      const last = matrix[i - 1][j];
      if (last) {
        let biggerValue = last[1] + values[i];
        // !!注意，如果位置上有东西，要进行比较，选择价值高的那个，这就是最重要的去重步骤
        // 为什么位置上会有东西呢？当前物品不拿的时候，延续了上一步的状态；
        // 在不涉及金额的版本里，这一步不用比较直接覆盖即可，也是去重的作用
        if (matrix[i][j + weights[i]])
          biggerValue = Math.max(biggerValue, matrix[i][j + weights[i]][1]);
        matrix[i][j + weights[i]] = [last[0] + weights[i], biggerValue];
      }
    }
  }
  // 矩阵填充完毕，从最后一行（做完全部选择且满足重量限制）的结果中，找出总价值最大的
  let res = 0;
  matrix[n - 1].forEach(obj => {
    res = Math.max(res, obj[1]);
  });
  return res;
}

// 上述问题中，事实上单元格保存当前价值即可，因为 column 已经记录了重量的信息
// 单元格保存 reference 的话，可以维持第三种以上的属性

// !根据 getMaxV DP 计算获得最大价值之后，可以倒推出具体的决策组合
// 思路：假设 res 对应的坐标是 [i,j] 其中 j 是重量
// 如果 [i-1, j] 存在，说明最后一个物品（i）不买，可以构成最佳组合
// 如果 [i-1, j - weights[i]] 存在，说明最后一个物品要买，可以构成最佳组合
// 循环检测，可以获得所有最佳组合

// 杨辉三角变种问题，每个节点有个数值，以及左右子节点
// 从最上层走到底部，每次可选择往左或往右，求最小和
// 思路：每次抉择只有两个选项，尝试使用状态转移表
// row 表示第几步，column 表示当前的和，上限是 9 * n（这里也可以用层高作为上限，更优）
// 单元格的元素需要是对象，表示当前的节点
function yanghuiTriangle(root) {
  const n = getHeight(root);
  const matrix = Array(n - 1).fill(Array(9 * n).fill(null)); // 因为第一层必选，所以只需要 N-1 个步骤
  matrix[0][root.left.val] = root.left; // 第一步，走左边
  matrix[0][root.right.val] = root.right; // 第一步，走右边
  // 从第二步开始，依据上一步，循环填充
  for (let i = 1; i < n - 1; i++) {
    // 做出选择：走左边（也可以理解为来自左边）
    for (let j = 0; j < 9 * n; j++) {
      const node = matrix[i - 1][j];
      if (node) {
        matrix[i][j + node.left.val] = node.left;
      }
    }
    // 做出选择：走右边（来自右边）
    // 有可能要去的位置上已经被刚才往左走的占领了，无所谓，覆盖ta，就是去重的概念
    for (let j = 0; j < 9 * n; j++) {
      const node = matrix[i - 1][j];
      if (node) {
        matrix[i][j + node.right.val] = node.right;
      }
    }
  }
  // 填充完毕，从最后一行倒序可得最大的和，正序可得最小的和，下标就是
  for (let index = 0; index < 9 * n; index++) {
    if (matrix[n - 2][index]) return index;
  }
}

// 辅助函数，求一棵树的高度
function getHeight(root) {
  if (!root) return 0;
  return 1 + Math.max(getHeight(root.left), getHeight(root.right));
}

// 对于二维矩阵-状态转移表来说，row 一般都用来表示第几步
// !只剩下 column 和单元格这两个地方能存放数据
// 对于只有一种属性的问题，用 column 表示累积值，单元格存储 Boolean 值即可
// 对于只有两种属性的问题，用 column 表示累积值，单元格存储第二种属性（累积）即可
// 对于三种以上属性的问题，单元格就需要存储 reference 对象了
// 属性是比较抽象的定义，比如限制了 i 和 j 的边界，那么 column 应该用来表示 j 坐标（看作属性），因为这种限制更精确
// 以上都是基于二维矩阵，要求每阶段的抉择选项最多只有 2 种

// 下面我们来看更复杂的状态转移方程解法
// !前提是：每次决策的选项超过 2 种
// 找零钱 w，有 1、3、5 三种硬币，求最少需要几个硬币
// 状态转移方程：cur = 1 + Math.min(...[payCoins(w - 5), payCoins(w - 3), payCoins(w - 1)]);
function payCoins(w, hashmap) {
  // 小于 0 的要被放弃，用无穷大作为哨兵
  if (w === 0) return 0;
  if (w < 0) return Infinity;
  // 哈希表防止重复计算
  if (w in hashmap) return hashmap[w];
  // ! 去重
  const cur =
    1 +
    Math.min(
      ...[
        payCoins(w - 5, hashmap),
        payCoins(w - 3, hashmap),
        payCoins(w - 1, hashmap)
      ]
    );
  hashmap[w] = cur;
  return cur;
}

// 比较字符串的相似程度，莱文斯坦距离（最短距离）
// 距离的大小表示差异的大小，可以增、删、替换字符
// 遇到相同字符，递归计算下一个字符，距离不变
// 遇到不同字符，选项有三种，距离加 1
function minDist(s, t, hashmap) {
  // 谁先结束，剩下的那个长度都是差异
  if (!s) return t.length;
  if (!t) return s.length;
  if (`${s},${t}` in hashmap) return hashmap[`${s},${t}`];
  if (s[0] === t[0]) {
    const cur = minDist(s.slice(1), t.slice(1));
    hashmap[`${s},${t}`] = cur;
    return cur;
  }
  const sLeft = s.slice(1);
  const tLeft = t.slice(1);
  // 去重的关键就是取最值
  const cur =
    1 +
    Math.min(...[minDist(s, tLeft), minDist(sLeft, tLeft), minDist(sLeft, t)]);
  hashmap[`${s},${t}`] = cur;
  return cur;
}

// 其实这种三个选项的情况，也可以用二维矩阵填充解决
// !关键在于定义 column 和单元格
// 这里可以用字符串 t 的下标作为 column，字符串 s 下标作为 row
// 单元格填充字符子串 s[:i] 和 t[:j] 的莱文斯坦距离
// 该距离来自上一个子串的距离推导
function minDistLoop(s, t) {
  const l1 = s.length,
    l2 = t.length,
    matrix = Array(l1).fill(Array(l2).fill(null));
  // 初始化第一行数据
  let hasI = false; // t[:j] 是否包含 s[0]，如果不包含距离就是 j + 1，否则 j
  for (let j = 0; j < l2; j++) {
    if (t[j] === s[0]) hasI = true;
    matrix[0][j] = hasI ? j : j + 1;
  }
  // 初始化第一列数据
  let hasJ = false;
  for (let i = 0; i < l1; i++) {
    if (s[i] === t[0]) hasJ = true;
    matrix[i][0] = hasJ ? i : i + 1;
  }
  // 从第二行和第二列开始填充
  for (let i = 1; i < l1; i++) {
    // 标准的做法应该分三次循环，每次都比较再覆盖
    // 这里直接借助转移方程，合并在一次循环里
    for (let j = 1; j < l2; j++) {
      if (s[i] === t[j]) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        // 当前状态来自于上一个状态的推导，有三种可能
        matrix[i][j] =
          1 +
          Math.min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]);
      }
    }
  }
  // 填充完毕，矩阵右下角就是答案
  return matrix[l1 - 1][l2 - 1];
}

// 字符串相似程度的另一种判断方式
// 最长公共子串长度，匹配方式只能增、删字符
// 与莱文斯坦距离的求解思路类似
function longestCommonSubStr(s, t, hashmap) {
  if (!s || !t) return 0;
  if (`${s},${t}` in hashmap) return hashmap[`${s},${t}`];
  if (s[0] === t[0]) {
    const cur = 1 + longestCommonSubStr(s.slice(1), t.slice(1));
    hashmap[`${s},${t}`] = cur;
    return cur;
  }
  const cur = Math.max(
    longestCommonSubStr(s.slice(1), t),
    longestCommonSubStr(s, t.slice(1))
  );
  hashmap[`${s},${t}`] = cur;
  return cur;
}

// 不同数字组成的数组，求最长递增子序列，不要求连续
// 回溯的思路是：以每个数字为起点，找出递增长度，最后取最长的那个
function lengthOfLIS(arr) {
  if (!arr.length) return 0;
  // 回溯，两种选项都尝试，选或者不选当前元素，
  const hashmap = {};
  function backTracking(i, prevNum) {
    if (i === arr.length) return 0;
    if (`${i},${prevNum}` in hashmap) return hashmap[`${i},${prevNum}`];
    // 不选
    let size1 = backTracking(i + 1, prevNum);
    // 选
    let size2 = 0;
    if (arr[i] > prevNum) {
      size2 = 1 + backTracking(i + 1, arr[i]);
    }
    // 这里其实已经是 DP 的思想了
    const cur = Math.max(size1, size2);
    hashmap[`${i},${prevNum}`] = cur;
    return cur;
  }
  return backTracking(0, -Infinity) || 1;
}

// 推荐 DP 的状态转移表
function lengthOfLISDP(arr) {
  // !定义子问题：f(i) 表示以 i 结尾的最大子序列长度
  // 状态转移方程：f(i) = arr(i) > arr(i-1) ? f(i-1) + 1 : f(i-1))，错误，这是连续子序列的状态转移
  // 正确的是：f(i) = arr(i) > arr(j) ? f(j) + 1 : f(j)，for j in [0,i]
  const n = arr.length,
    dp = Array(n).fill(0);
  if (!n) return 0;
  let res = 1;
  dp[0] = 1;
  for (let i = 1; i < n; i++) {
    let prevMax = 0;
    for (let j = 0; j < i; j++) {
      if (arr[i] > arr[j]) {
        // !只需要取 j 能达到的最大长度
        prevMax = Math.max(prevMax, dp[j]);
      }
    }
    dp[i] = 1 + prevMax;
    res = Math.max(res, dp[i]);
  }
  return res;
}

// TODO 上述问题最强解法是 DP + 二分查找

// 求 subArray max sum
var maxSubArray = function(nums) {
  // 尝试 DP 的状态转移表，一维数组
  // 这里的 n 个步骤指的是，求以 nums[n] 结尾的子序列的最大和
  // dp[i] 表示以 nums[i] 结尾的子序列，其值为该子序列的 maxSum
  // 如果 dp[i-1] < 0，则 dp[i] = nums[i]，否则 dp[i] = dp[i-1] + nums[i]
  // 状态转移方程：dp[i] = max(nums[i], dp[i-1] + nums[i])
  const n = nums.length,
    dp = Array(n).fill(null);
  dp[0] = nums[0]; // 初始化第一个元素
  for (let i = 1; i < n; i++) {
    // 其实也算两种选择，要还是不要前面的子序列
    dp[i] = Math.max(nums[i], nums[i] + dp[i - 1]);
  }
  return Math.max(dp);
};

// 递归解法，借助状态转移方程
function maxSubArrayRecur(nums) {
  let maxSum = -nums[0];
  // 辅助函数
  function recur(nums, i) {
    if (i === 0) {
      return nums[0];
    }
    const lastSum = recur(nums, i - 1);
    const curSum = lastSum > 0 ? lastSum + nums[i] : nums[i];
    // 顺便操作外部变量
    if (curSum > maxSum) maxSum = curSum;
    return curSum;
  }
  recur(nums, nums.length - 1);
  return maxSum;
}

// 求字符串的最长回形子串
var longestPalindrome = function(s) {
  // 定义子问题：f(i,j) 标记 i - j 是否组成回形字符串
  // 如果 f(i+1,j-1) == true 且 s[i] == s[j]，那么 f(i,j) = true
  // 起始条件 f(i,i) = true 或者 f(i,i+1) = true if s[i] == s[i+1]
  // 状态转移表，二维数组
  // 直觉一：遍历 s，以每个字符为中心，尝试向两边扩张填表
  // 直觉二：遍历 s，以每个字符为右边界，向左边扩张填表
  // !相比之下，第二种方式比较好填表
  const n = s.length;
  const dp = Array.from(Array(n), () => new Array(n));
  let maxL = 0,
    res = "";
  // 以字符 j 为右边界，每次嵌套循环检测左边界，填表
  for (let j = 0; j < n; j++) {
    for (let i = j; i > -1; i--) {
      // 这里的 j - i < 3 也是神来之笔，简化了不少代码
      dp[i][j] = s[i] === s[j] && (j - i < 3 || dp[i + 1][j - 1]);
      if (dp[i][j] && j - i + 1 > maxL) {
        maxL = j - i + 1;
        res = [i, j];
      }
    }
  }
  return s.slice(res[0], res[1] + 1);
};

// !此题的最强解法：遍历每个点作为中点，往两边扩张
// 类似状态转移表法，但只需要指针记录最值，不需要二维数组
function longestPalindromeBest(s) {
  if (!s) return "";
  let n = s.length,
    maxL = -1,
    res = [0, 0];
  for (let i = 0; i < n; i++) {
    tryPalindrome(s, i, i);
    tryPalindrome(s, i, i + 1);
  }
  // 辅助函数，从中点往两边扩张，并更新当前最大值
  function tryPalindrome(s, i, j) {
    while (i >= 0 && j < s.length && s[i] === s[j]) {
      i--;
      j++;
    }
    if (j - i + 1 > maxL) {
      maxL = j - i + 1;
      res = [i + 1, j];
    }
  }
  return s.slice(res[0], res[1]);
}

// 正则匹配，支持 . 和 *
// 其中 * 只能跟在其他字符后面，表示任意个该字符（并非完全自由）
var isMatch = function(s, p) {
  // 定义子问题：f(i,j) 表示 s[i:] 和 p[j:] 是否匹配
  // ! 如果 j+1 是 *，我们有两种选择：忽略 j 和 j+1，或者确定 i 与 j 匹配之后，下一步继续用 i+1 与 j 匹配（因为 j* 可以无限）
  // 如果 j+1 不是 *，只需 i 与 j 匹配之后，递归 i+1 与 j+1
  // 终止条件是 p 为空时，s 为空
  const n = s.length,
    m = p.length,
    cache = Array.from(Array(n + 1), () => new Array(m + 1).fill(null));
  function recur(i, j) {
    if (j === m) return i === n;
    if (cache[i][j] !== null) return cache[i][j];
    const firstMatch = i < n && [s[i], "."].includes(p[j]);
    // j+1 星号，两种选择
    if (j <= m - 2 && p[j + 1] === "*") {
      const ignoreP = recur(i, j + 2);
      const ignoreS = firstMatch && recur(i + 1, j);
      const cur = ignoreP || ignoreS;
      cache[i][j] = cur;
      return cur;
    }
    const cur = firstMatch && recur(i + 1, j + 1);
    cache[i][j] = cur;
    return cur;
  }
  return recur(0, 0);
};

// 正则匹配第二版，支持 ? 和 *
// 其中 ? 与上题的 . 一样，* 则更强大，能表示任意字符串
var isMatch2 = function(s, p) {
  // 与第 10 题类似，但是这里的 * 拥有更强大的匹配能力
  // 定义子问题：dp(i,j) 表示 s[:i-1] 和 p[:j-1] 能否匹配
  // ! 如果 p[j-1] === * 那么 dp[i,j] = dp[i-1,j] || dp[i,j-1]
  // 如果 p[j-1] !== * 那么 dp[i,j] = p[j-1] in [s[i-1], '?'] && dp[i-1,j-1]
  // ? tricky part: (i,j) 只能表示截止 i-1,j-1 的匹配，否则会遗漏某些东西
  const n = s.length,
    m = p.length,
    dp = Array.from(Array(n + 1), () => new Array(m + 1));
  // 初始化第一行和第一列
  dp[0][0] = true;
  for (let i = 1; i <= m; i++) {
    if (p[i - 1] === "*") {
      dp[0][i] = true;
    } else {
      break;
    }
  }
  // 开始填充
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      // 去重的过程
      if (p[j - 1] === "*") {
        dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
      } else {
        dp[i][j] = dp[i - 1][j - 1] && [s[i - 1], "?"].includes(p[j - 1]);
      }
    }
  }
  return !!dp[n][m];
};

// 求最长连续有效括号
var longestValidParentheses = function(s) {
  // 直觉：stack，左括号入栈，右括号与栈顶配对，出栈
  // 每次能获得一次连续的括号长度，尝试更新
  // 但是多余的左括号会造成污染，无法判断连续
  // 需要记住所有括号的下标，这种解法比 DP 优秀
  let res = 0,
    stack = [-1]; // 初始的 -1 下标非常重要，应对左侧完美的情况
  for (let i in s) {
    const k = s[i];
    if (k === "(") {
      stack.push(i);
    } else {
      stack.pop();
      if (!stack.length) {
        // !tricky part: 最左侧的 ）位置标记，是计算连续长度的关键
        stack.push(i);
      }
      const last = stack[stack.length - 1];
      if (i - last > res) res = i - last;
    }
  }
  return res;
};

// 加密 A -> 1, Z -> 26, 给定一个长数字
// 求总共有多少种解密方式（注意 0 无法单独解密）
var numDecodings = function(s) {
  // 定义子问题：f(i) 表示截止 s[i-1] 能组成的解密方式
  // 如果最后两位组成的数字在 10 到 26 之间，则 f(i) 至少拥有 f(i-2) 的解法数量
  // 如果最后一位数字不是 0，则 f(i) 至少拥有 f(i-1) 的解法数量
  // !本题的难点在于 0 的处理，以及初始化 dp[0] = 1 为了后面计算使用
  // 状态转移表，发现了没，又是 n+1 位
  const n = s.length,
    dp = Array(n + 1).fill(null);
  dp[0] = 1;
  dp[1] = s[0] === "0" ? 0 : 1;
  for (let i = 2; i < n + 1; i++) {
    const last = s[i - 1];
    const lastTwo = Number(s[i - 2] + s[i - 1]);
    if (last !== "0") dp[i] += dp[i - 1];
    if (lastTwo >= 10 && lastTwo <= 26) dp[i] += dp[i - 2];
  }
  return dp[n];
};

// 给定正整数 1...n，列出能生成的所有 BST
var generateTrees = function(n) {
  // 取任意 i 为 root，那么 1...i-1 肯定是左树，i+1...n 肯定是右树
  // 天然递归的过程，分治思想，而不是 DP
  // ! 返回 start 到 end 之间能组成的 BST 数组
  function recur(start, end) {
    if (start > end) return [null];
    const arr = [];
    for (let i = start; i <= end; i++) {
      let leftArr = recur(start, i - 1);
      let rightArr = recur(i + 1, end);
      for (let left of leftArr) {
        for (let right of rightArr) {
          const node = new TreeNode(i);
          node.left = left;
          node.right = right;
          arr.push(node);
        }
      }
    }
    return arr;
  }
  if (!n) return [];
  return recur(1, n);
};

// 上题的变种：只求数量，那么就可以用缓存

// 给定字符串 s1, s2, s3, 判断 s3 能否由前两者交叉组成，字符的相对顺序不能变
var isInterleave = function(s1, s2, s3) {
  // 直觉：将 s1 和 s2 所有字符统计到数组里
  // 遍历 s3 扣除数组相应下标数量，最后检查数组是否为空
  // 但是要求相对顺序不能变，因此上述解法无效
  // 定义子问题：f(i,j,k) 表示 s1[:i] s2[:j] 能组成 s3[:k]
  // !s1 尾巴或 s2 尾巴与 s3 尾巴相等
  // f(i,j,k) = f(i-1,j,k-1) && s1[i] === s3[k] || f(i,j-1,k-1) && s2[j] === s3[k]
  // 终止条件：f(-1,j,k) = s2[:j] === s3[:k], f(i,-1,k) = s1[:i] === s3[:k], f(-1,-1,-1) = true;
  // 这种情况下不好用状态转移表（超过二维），直接用递归 + 字符串哈希表
  const l1 = s1.length,
    l2 = s2.length,
    l3 = s3.length;
  const cache = {};
  function recur(i, j, k) {
    if (i === -1) return s2.slice(0, j + 1) === s3.slice(0, k + 1);
    if (j === -1) return s1.slice(0, i + 1) === s3.slice(0, k + 1);
    if (k === -1) return i === -1 && j === -1;
    if (`${i},${j},${k}` in cache) return cache[`${i},${j},${k}`];
    const cur =
      (recur(i - 1, j, k - 1) && s1[i] === s3[k]) ||
      (recur(i, j - 1, k - 1) && s2[j] === s3[k]);
    cache[`${i},${j},${k}`] = cur;
    return cur;
  }
  return recur(l1 - 1, l2 - 1, l3 - 1);
};

// 字符拆分，给定一个字符串和一个数组，判断字符串能否拆分成数组的成员
var wordBreak = function(s, wordDict) {
  // 定义子问题：f(i) 表示 s[i:] 是否合格
  // 找出数组中最长和最短的词的长度 minW MaxW
  // 如果 f(i) 合格，说明右侧递进到 x，肯定存在 x-i in set && f(x) === true
  // !这里的状态转移方程不再是与 f(i-1) 的关系，而是跳跃式的
  // f(i) = f(i+minW) && s[i,i+minW] in set || f(i+minW+1) && ... || ... || f(i+maxW) && ...
  // 初始条件 f(n) = true
  // 状态转移表
  const set = new Set(wordDict);
  const n = s.length;
  let minW = Infinity,
    maxW = -Infinity;
  wordDict.forEach(w => {
    const curLen = w.length;
    if (curLen > maxW) maxW = curLen;
    if (curLen < minW) minW = curLen;
  });
  const dp = Array(n + 1).fill(null);
  // 初始化
  dp[n] = true;
  for (let i = n - 1; i > -1; i--) {
    let temp = false;
    // 检查 i+minW 到 i+maxW 之间是否存在匹配的
    for (let j = i + minW; j <= i + maxW && j <= n; j++) {
      if (dp[j] && set.has(s.slice(i, j))) {
        temp = true;
        break;
      }
    }
    dp[i] = temp;
  }
  return dp[0];
};

// 求 subArray 最大乘积
var maxProduct = function(nums) {
  // 与 max sum subArray 问题类似，但有一个陷阱：当前的负数，后面可能会翻身
  // 最大值的获得，可能是最小的负数 * 当前负数，也可能是最大正数 * 当前正数
  // 我们让单元格保存 reference，分别是以 ith 结尾的最大、最小乘积
  // f(i)[0] = min(f(i-1)[0] * nums[i], f(i-1)[1] * nums[i], nums[i])
  // f(i)[1] = max(f(i-1)[0] * nums[i], f(i-1)[1] * nums[i], nums[i])
  // 初始条件：f(0) = [nums[0], nums[0]]
  const n = nums.length,
    dp = Array.from(Array(n), () => new Array(2));
  let res = nums[0];
  dp[0] = [nums[0], nums[0]];
  for (let i = 1; i < n; i++) {
    dp[i][0] = Math.min(
      dp[i - 1][0] * nums[i],
      dp[i - 1][1] * nums[i],
      nums[i]
    );
    dp[i][1] = Math.max(
      dp[i - 1][0] * nums[i],
      dp[i - 1][1] * nums[i],
      nums[i]
    );
    if (dp[i][1] > res) res = dp[i][1];
  }
  return res;
};

// house robber 加强版，首尾相连不能同时抢，邻居不能同时抢
var rob = function(nums) {
  // 定义子问题：f(i) 表示截止 ith 能抢到的最大值
  // 普通版的状态转移方程：f(i) = max(f(i-1), f(i-2) + nums[i])
  // 头尾相连我们可以分为两部分，(0,n-2) 以及 (n-1,1)
  // 使用两个状态转移表，都是一维数组
  const n = nums.length;
  if (!n) return 0;
  if (n <= 2) return Math.max(...nums);
  const dp1 = Array(n).fill(null),
    dp2 = Array(n).fill(null);
  dp1[0] = nums[0];
  dp1[1] = Math.max(nums[0], nums[1]);
  dp2[0] = nums[n - 1];
  dp2[1] = Math.max(nums[n - 1], nums[n - 2]);
  for (let i = 2; i < n - 1; i++) {
    dp1[i] = Math.max(dp1[i - 1], dp1[i - 2] + nums[i]);
    dp2[i] = Math.max(dp2[i - 1], dp2[i - 2] + nums[n - 1 - i]);
  }
  // !最后从四种结尾找出最大的
  return Math.max(dp1[n - 2], dp1[n - 3], dp2[n - 2], dp2[n - 3]);
};

// 二维矩阵只包含 '0' 和 '1'
// !求 '1' 组成的最大正方形面积
var maximalSquare = function(matrix) {
  // 定义子问题：f(i,j) 表示以 [i,j] 作为右下角的最大正方形边长
  // 如果 matrix[i][j] === 1，那么 f(i,j) = min(f(i-1,j),f(i,j-1),f(i-1,j-1)) + 1
  // '1' 要看作最小正方形
  if (!matrix.length) return 0;
  const m = matrix.length,
    n = matrix[0].length;
  let res = 0;
  const dp = Array.from(Array(m), () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    dp[i][0] = matrix[i][0] === "1" ? 1 : 0;
    if (dp[i][0] > res) res = dp[i][0];
  }
  for (let j = 0; j < n; j++) {
    dp[0][j] = matrix[0][j] === "1" ? 1 : 0;
    if (dp[0][j] > res) res = dp[0][j];
  }

  // 开始填充
  for (let i = 1; i < m; i++) {
    for (let j = 1; j < n; j++) {
      if (matrix[i][j] === "1") {
        dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1;
        if (dp[i][j] > res) res = dp[i][j];
      }
    }
  }
  return res ** 2;
};

// ugly number 进阶，只能被 2、3、5 整除，求第 n 个 ugly number
// 1 是 ugly number
var nthUglyNumber = function(n) {
  // 维护一个额外的 set，存储前 i-1 个 un
  // 从 set[-1] + 1 开始递增，如果 num/2 或 num/3 或 num/5 的商属于 set
  // 那么 f(i) = num；set.add(num)
  const set = new Set([1]);
  let i = 1,
    k = 1;
  while (i < n) {
    // ? 逐个递增太慢
    k += 1;
    if (set.has(k / 2) || set.has(k / 3) || set.has(k / 5)) {
      i += 1;
      set.add(k);
    }
  }
  return k;
};

// 接下来看看巧妙的 DP 解法
// 定义子问题：f(i) 表示第 i 个目标数字
// 这里的 f(i) 无法简单由 f(i-1) 等递推，但是后面的数只能由前面的数乘以 2、3、5 获得
// 我们额外维护三个变量 t2,t3,t5
// f(i) 就由这三个变量推导获得：f(i) = min(2 * f(t2), 3 * f(t3), 5 * f(t5))
// 一旦确定当前最小值，对应的变量 tx += 1，然后继续上述循环
function nthUglyNumberDP(n) {
  if (!n) return;
  let t2 = 0,
    t3 = 0,
    t5 = 0,
    dp = Array(n).fill(0);
  dp[0] = 1;
  let i = 1;
  while (i < n) {
    // !这一步就是去重
    dp[i] = Math.min(2 * dp[t2], 3 * dp[t3], 5 * dp[t5]);
    // 这三步就看作状态推导的过程
    if (dp[i] === 2 * dp[t2]) t2++;
    if (dp[i] === 3 * dp[t3]) t3++;
    if (dp[i] === 5 * dp[t5]) t5++;
    i++;
  }
  return dp[n - 1];
}
