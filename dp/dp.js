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
  // !每个单元格我们存储一个二元数组，表示当前组合的价值和重量（稍微过度设计）
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

console.log(getMaxV());

// 上述问题中，事实上单元格保存当前价值即可，因为 column 已经记录了重量的信息
// 单元格保存 reference 的话，可以维持第三种以上的属性了

// !根据 getMaxV DP 计算获得最大价值之后，可以倒推出具体的决策组合
// 思路：假设 res 对应的坐标是 [i,j]
// 如果 [i-1, j] 存在，说明最后一个物品（i）不买，可以构成最佳组合
// 如果 [i-1, res - values[i]] 存在，说明最后一个物品要买，可以构成最佳组合
// 循环检测，可以获得所有最佳组合

// 杨辉三角变种问题，每个节点有个数值，以及左右子节点
// 从最上层走到底部，每次可选择往左或往右，求最小和
// 思路：每次抉择只有两个选项，尝试使用状态转移表
// row 表示第几步，column 表示当前的和，上限是 9 * n
// 单元格的元素需要是对象，表示当前的节点
function yanghuiTriangle(root) {
  const n = getHeight(root);
  const matrix = Array(n - 1).fill(Array(9 * n).fill(null)); // 因为第一层必选，所以只需要 N-1 个步骤
  matrix[0][root.left.val] = root.left; // 第一步，走左边
  matrix[0][root.right.val] = root.right; // 第一步，走右边
  // 从第二步开始，依据上一步，循环填充
  for (let i = 1; i < n - 1; i++) {
    // 做出选择：走左边
    for (let j = 0; j < 9 * n; j++) {
      const node = matrix[i - 1][j];
      if (node) {
        matrix[i][j + node.left.val] = node.left;
      }
    }
    // 做出选择：走右边
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
  if (w === 0) return 1;
  if (w < 0) return Infinity;
  // 哈希表防止重复计算
  if (w in hashmap) return hashmap[w];
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
