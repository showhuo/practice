// 给定数字 n，生成所有合法的括号组合
var generateParenthesis = function(n) {
  // !有约束条件的回溯
  const res = [];
  function recur(i, tempStr, leftCount, rightCount) {
    if (i === 2 * n) {
      res.push(tempStr);
      return;
    }
    // 选左括号
    if (leftCount < n) {
      recur(i + 1, tempStr + "(", leftCount + 1, rightCount);
    }
    // 选右括号
    if (rightCount < leftCount) {
      recur(i + 1, tempStr + ")", leftCount, rightCount + 1);
    }
  }
  recur(0, "", 0, 0);
  return res;
};

// 模拟 indexOf 的实现
var strStr = function(haystack, needle) {
  if (needle === "") return 0;
  // 两个指针，如果 needle 能走完，说明匹配
  let i = 0,
    j = 0,
    n = haystack.length,
    m = needle.length,
    res = -1;
  while (i < n && j < m) {
    if (haystack[i] !== needle[j]) {
      // !tricky part: 如果部分匹配，不能直接跳 i+1，反而要回去 res+1 再开始
      i = res === -1 ? i + 1 : res + 1;
      // tricky part: 一旦不匹配，重置 j
      j = 0;
      res = -1;
    } else {
      // 只能标记第一次匹配
      if (res === -1) res = i;
      i++;
      j++;
    }
  }
  // 如果 j 走完了，说明匹配
  return j === m ? res : -1;
};

// 将二维矩阵右旋90度
var rotate = function(matrix) {
  // !脑筋急转弯：右旋90度相当于先上下翻转 --> 以左上对角线对称翻转
  // 只要是以左上对角线对称翻转，必然有 [i,j] = [j,i]，因为原点就是左上角
  // 同理可得，左旋90度相当于先左右翻转，再对称翻转
  const n = matrix.length;
  let i = 0,
    j = n - 1;
  while (i < j) {
    [matrix[i], matrix[j]] = [matrix[j], matrix[i]];
    i++;
    j--;
  }
  for (let k = 0; k < n; k++) {
    // tricky part: 只能操作一半，即右上部分
    for (let h = k + 1; h < matrix[k].length; h++) {
      [matrix[k][h], matrix[h][k]] = [matrix[h][k], matrix[k][h]];
    }
  }
};

// 给定二维矩阵，要求顺时针回旋打印
var spiralOrder = function(matrix) {
  // !这个回旋收缩需要 4 个边界和 4 个 loop
  // 每次 loop 收缩其中一个边界
  if (!matrix.length) return [];
  const m = matrix.length,
    n = matrix[0].length,
    res = [];
  let top = 0,
    bottom = m - 1,
    left = 0,
    right = n - 1;
  while (bottom >= top && left <= right) {
    for (let i = left; i <= right; i++) {
      res.push(matrix[top][i]);
    }
    top++;
    for (let i = top; i <= bottom; i++) {
      res.push(matrix[i][right]);
    }
    right--;
    // tricky part: 经由上面两个 loop，可能边界已经全部收缩完毕，因此需要额外检查
    if (bottom >= top) {
      for (let i = right; i >= left; i--) {
        res.push(matrix[bottom][i]);
      }
      bottom--;
    }
    if (left <= right) {
      for (let i = bottom; i >= top; i--) {
        res.push(matrix[i][left]);
      }
      left++;
    }
  }
  return res;
};

// 给定二叉树的前序、中序遍历数组，要求还原二叉树
var buildTree = function(preorder, inorder) {
  // 这题挺难的，需要非常熟悉前序、中序等数组特点
  // preorder[0] 肯定是 root，先找出 root 在 inorder 数组的下标 idx
  // inorder 的 idx 左边就是左树，右边是右树
  // TODO 为了避免每次找下标 idx，可以先构造一个哈希表用 O(1) 取下标
  // const hashmap = {};
  // for (let i = 0; i < inorder.length; i++) {
  //   hashmap[inorder[i]] = i;
  // }
  function recur(preorder, inorder) {
    if (!inorder || !inorder.length) return null;
    const val = preorder.shift();
    const root = new TreeNode(val);
    const idx = inorder.indexOf(val);
    root.left = recur(preorder, inorder.slice(0, idx));
    root.right = recur(preorder, inorder.slice(idx + 1));
    return root;
  }
  return recur(preorder, inorder);
};

// 给定二维矩阵和一个字符串，判断字符串能否由矩阵的横竖连字符组成
var exist = function(board, word) {
  // 如果 word 是一次性判断，回溯矩阵
  // 如果 word 是动态且大量的，那么需要先将矩阵转化成 Trie 树
  // 这里先演示第一种解法
  let m = board.length,
    n = board[0].length,
    wLen = word.length;

  function recur(i, j, k) {
    if (k === wLen) return true;
    if (i < 0 || i > m - 1 || j < 0 || j > n - 1) {
      return false;
    }
    // 当前字符不匹配
    if (board[i][j] !== word[k]) return false;
    //! 已经访问过，巧妙的污染，省略了大量的哈希表判重
    const temp = board[i][j];
    board[i][j] = "#";
    const res =
      recur(i - 1, j, k + 1) ||
      recur(i + 1, j, k + 1) ||
      recur(i, j - 1, k + 1) ||
      recur(i, j + 1, k + 1);
    board[i][j] = temp;
    return res;
  }

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (board[i][j] === word[0]) {
        if (recur(i, j, 0)) return true;
      }
    }
  }
  return false;
};
