// 排1和2，快排的分区思路
function sortTwoNums(arr) {
  let i = 0,
    j = 0;
  while (j < arr.length) {
    if (arr[j] === 1) {
      [arr[i], arr[j]] = [arr[j], arr[i]];
      i += 1;
    }
    j += 1;
  }
  return arr;
}
// console.log(sortTwoNums([2, 1, 1, 2, 1, 1, 2]));

// 机器人从0开始，可以走一步，也可以走当前位置的两倍，求走到 dest 的最少步数
// 应该让最后几步尽可能翻倍，因为跨度大，倒推，转化为尽量整除 2
function leastSteps(dest) {
  res = 0;
  while (dest > 1) {
    if (dest % 2 === 0) {
      dest /= 2;
      res += 1;
    } else {
      dest = (dest - 1) / 2;
      res += 2;
    }
  }
  return res + 1;
}
// console.log(leastSteps(500));

// 递归
function leastStepsRecur(dest, res) {
  if (dest === 1) {
    return res + 1;
  }
  if (dest % 2 === 0) {
    newDest = dest / 2;
    res += 1;
    return leastStepsRecur(newDest, res);
  } else {
    newDest = (dest - 1) / 2;
    res += 2;
    return leastStepsRecur(newDest, res);
  }
}
// console.log(leastStepsRecur(500, 0));

// 构造函数使得 f(a,b) = f(a)(b) = a + b
// 函数式编程
// 同理可以扩展为无数个参数
function f(...args) {
  if (args.length === 1) {
    return b => {
      return args[0] + b;
    };
  } else if (args.length === 2) {
    return args[0] + args[1];
  } else {
    throw new Error("f needs 1 or 2 arguments.");
  }
}
// let a = f(1);
// console.log(a(10));

// 写一个柯里函数，使得它在 fn() 执行前可以不断累加参数值
// 比如 fn(1)(2,3)(4,5,6)() === 21
function currySum(...args) {
  let cache = args;
  return function newFn(...newArgs) {
    if (newArgs.length === 0) {
      return cache.reduce((a, b) => a + b);
    }
    cache = cache.concat(newArgs);
    return newFn;
  };
}
// let fn = currySum(1)(2, 3);
// console.log(fn(4, 5, 6)());

// 下一步可以引申出写一个 curry 函数，将普通函数 curry 化

// 数组中大部分数字都出现 2 次，只有一个数字出现一次，找出它
// 任何变量异或 ^ 自己都等于 0
function findUniNum(arr) {
  let res = 0;
  for (const ele of arr) {
    res ^= ele;
  }
  return res;
}

// 假值判断
function judgeFake() {
  if ([] == false) console.log(1); // yes
  if ([]) console.log(3); // yes
  if ({} == false) console.log(2);
  if ({}) console.log(5); // yes
  if ([1] == [1]) console.log(4);
}

// 异步执行
// nextTick >> promise.then >> await 下一行 >> setTimeout
function asyncOrder() {
  async function a1() {
    console.log("a1 start"); // 2
    await a2();
    console.log("a1 end"); // ! 倒数第二个，在 settimeout 之前
  }
  async function a2() {
    console.log("a2"); // 3
  }

  console.log("script start"); // 1

  setTimeout(() => {
    console.log("setTimeout"); // last
  }, 0);

  Promise.resolve().then(() => {
    console.log("promise1"); // 7
  });

  process.nextTick(() => console.log("next tick!"));

  a1();

  let promise2 = new Promise(resolve => {
    resolve("promise2.then");
    console.log("promise2"); // 5
  });

  promise2.then(res => {
    console.log(res); // 8
    Promise.resolve().then(() => {
      console.log("promise3"); // 9
    });
  });
  console.log("script end"); // 6
}
// asyncOrder();

// this
function thisQ() {
  const obj = {
    a: () => console.log(this), // {}
    b: function() {
      console.log(this);
    }
  };
  obj.a();
  obj.b();
}

// let
function letP(params) {
  const obj = {
    name: "name",
    skill: ["es6", "react", "nodejs"],
    say: function() {
      for (let i = 1; i <= obj.skill.length; i++) {
        setTimeout(function() {
          console.log("No." + i + obj.name);
          console.log(obj.skill[i - 1]);
          console.log("----------");
        }, 0);
        console.log(i);
      }
    }
  };
  obj.say();
}

// 节流函数 throttle 简易版
function throttle(fn, interval) {
  let timer = false;
  return (...args) => {
    if (timer) return; // 拒绝执行
    timer = true;
    fn(...args);
    setTimeout(() => {
      timer = false;
    }, interval);
  };
}

// 还有另一种增强版，确保最后一次调用一定会被执行，需要 Date.now 计算剩余时间
// 相当于递进式的 debounce，剩余时间不断缩小
const throttle2 = (func, limit) => {
  let timer;
  let lastRan;
  return function() {
    const context = this;
    const args = arguments;
    if (!lastRan) {
      func.apply(context, args);
      lastRan = Date.now();
    } else {
      clearTimeout(timer);
      timer = setTimeout(function() {
        if (Date.now() - lastRan >= limit) {
          func.apply(context, args);
          lastRan = Date.now();
        }
      }, limit - (Date.now() - lastRan));
    }
  };
};

// 压制函数 debounce，重新触发就重置计时器
function debounce(fn, wait) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      fn(...args);
    }, wait);
  };
}

// 订阅发布模式
class PubSub {
  constructor(name) {
    this.name = name;
    this.listeners = {};
  }
  subScribe(event, callback) {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);
  }
  publish(event, data) {
    if (!this.listeners[event] || !this.listeners[event].length) {
      return;
    }
    this.listeners[event].forEach(cb => {
      cb(data);
    });
  }
  // 取消订阅
  unSubscribe(event) {
    delete this.listeners[event];
  }
}

// leetcode
// 求不重复的最长连续字符串，双指针 sliding window
function lengthOfLongestSubstring(s) {
  const hashmap = {};
  let i = 0,
    j = 0,
    len = s.length,
    res = 0;
  while (j < len) {
    let w = s[j];
    if (!(w in hashmap)) {
      hashmap[w] = j;
      j += 1;
      res = Math.max(res, j - i);
    } else {
      // i 跳到 hashmap[w] 之间的字母需要从哈希表中删除
      for (let index = i; index < hashmap[w]; index++) {
        delete hashmap[s[index]];
      }
      i = hashmap[w] + 1;
      hashmap[w] = j;
      j += 1;
    }
  }
  return res;
}

// 求一组字符串的最长公共前缀
function longestCommonPrefix(strs) {
  for (let i = 0; i < strs[0].length; i++) {
    const w = strs[0][i];
    for (let j = 1; j < strs.length; j++) {
      if (strs[j].length < i + 1 || strs[j][i] !== w) {
        return strs[0].slice(0, i);
      }
    }
  }
  return strs[0];
}

// 字符串s1的某种排列，是否是字符串s2的子串
// 思路：
// 因为 s1 顺序无法预测，只能遍历 s2，用 s1 构造哈希表作为剩余可匹配的字符集
// 双指针构成区间 [i, j]，如果 j 遇到不是 s1 的字符，i 需要重置到 j 的位置
// 如果 s1 有重复字符，稍微麻烦一点，其哈希表需要记录个数
// 如果 j 遇到 s1 的字符，但是哈希表中该字符的剩余个数为 0，i 需要重置到第一个 j 字符位置的下一位
// 因此要求 s1 哈希表还要记录下标，所以 s1 哈希表需要使用队列来记录 key 对应的下标们（既有下标，又有个数）
// 当 i 指针发生跳跃重置，当前使用的 s1 哈希表也需要重置
// 如果是 j 遇到非法字符的情况，重新复制一份原始 s1 哈希表
// 如果是 j 遇到合法字符，但是 i 需要移位的情况，将当前哈希表对应 i - 新位置之间的字符的下标数组们，补回一些库存
// 但是呢，补回下标库存的操作逻辑相当麻烦，我们需要重新审视 i 跳跃下标的问题，事实上从 i 往 j 逐步检查，也能找到第一个 j 字符的位置
// 也就是说 s1 哈希表记录个数就足够了
function checkInclusion(s1, s2) {
  const lengthS1 = s1.length;
  const hashmap = {}; // 存储 s1 的所有字符对应的下标队列
  for (const w of s1) {
    if (!(w in hashmap)) {
      hashmap[w] = 0;
    }
    hashmap[w] += 1;
  }
  // 遍历 s2
  let i = 0,
    j = 0;
  let copyHashmap = { ...hashmap };
  while (j < s2.length) {
    let w = s2[j];
    if (w in copyHashmap) {
      if (copyHashmap[w] >= 1) {
        copyHashmap[w] -= 1;
        // 如果这时候已经匹配成功，直接return
        if (j - i + 1 === lengthS1) return true;
        j += 1;
      } else {
        // 字符个数剩余 0，需要将 i 往右移到第一个 j 字符位置的下一位，期间补回中途字符的个数
        while (i < j) {
          if (s2[i] !== w) {
            copyHashmap[s2[i]] += 1; // 补回一位
            i += 1;
          } else {
            break;
          }
        }
        // 找到 j 字符，i 再右移一位，j 字符个数刚好抵消，不需要额外操作
        i += 1;
        j += 1;
      }
    } else {
      // 不存在的字母，则重置 copyHashmap
      copyHashmap = { ...hashmap };
      j += 1;
      i = j;
    }
  }
  // 遍历完之后，检查长度是否满足
  return j - i === lengthS1;
}
// console.log(checkInclusion("ky", "ainwkckifykxlribaypk"));

// 字符串相乘，不能直接使用大数字
// 思路：
// 123 * 456 === （100 * 1 + 2 * 10 + 3） * （4 * 100 + 5 * 10 + 6）
// 两个for循环计算然后累加
// 因为不能直接使用大数字，只能每次取一位相乘，用数组下标移位累加结果
// 巧妙的是下标 i * j 的结果只会被累加在 res[i+j] 和 res[i+j+1] 这两个位置
function multiply(num1, num2) {
  if (num1 === "0" || num2 === "0") return "0";
  let m = num1.length,
    n = num2.length;
  const tempArr = Array(m + n).fill(0); // 位置刚好
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      let mul = Number(num1[i]) * Number(num2[j]);
      let sum = mul + tempArr[i + j + 1];
      tempArr[i + j + 1] = sum % 10;
      // 高位超过10的话也无所谓
      tempArr[i + j] += Math.floor(sum / 10);
    }
  }
  // tempArr 空间刚好，但首位的 0 可以跳过
  let res = "";
  let i = 0;
  if (tempArr[0] === 0) i += 1;
  while (i < tempArr.length) {
    res += tempArr[i];
    i += 1;
  }
  return res;
}
// console.log(multiply("2", "3"));

// 反转字符串并去除多余空格
function reverseWords(s) {
  const arr = s.split(" ").filter(Boolean);
  let res = "";
  for (let i = 0; i < arr.length; i++) {
    res = arr[i] + " " + res;
  }
  return res.slice(0, res.length - 1);
}

// 简化路径，Unix 风格的三种路径符号 / . .. 将其标准化，去除多余的 / 和 .
// str.split('/') 然后遍历数组处理
// 后来的..可以抵消前面的字母，很适合 stack
function simplifyPath(path) {
  const arr = path.split("/");
  const stack = [];
  let res = "";
  for (const w of arr) {
    if (w == "..") {
      stack.pop();
    } else if (/\w/.test(w)) {
      stack.push(w);
    } else if (w.includes("..")) {
      // 3个以上的.连在一起的情况，拆成 .. 剩余的单个 . 忽略
      let numOfdd = Math.floor(w.length / 2);
      while (numOfdd > 0) {
        stack.pop();
        numOfdd -= 1;
      }
    }
  }
  if (stack.length) res += stack.pop(); // 尾巴
  while (stack.length) {
    res = stack.pop() + "/" + res;
  }
  return "/" + res;
}

// 给定一个字符串，拆成所有可能的 IP 地址
// 问题等价于将字符串拆成4部分，每部分数字范围在 0-255
function restoreIpAddresses(s) {
  // 初步考虑回溯
  // 剩余的长度上限要限制，比如剩余2位，则长度不能大于6
  let res = new Set();
  // 回溯
  function backTracking(s, count, minSize, tempArr) {
    if (!s) {
      if (count === 4) res.add(tempArr.join("."));
      return;
    }
    if (s.length > (4 - count) * 3 || s.length < (4 - count) * minSize) return;
    // 一次可以切割 minSize 到 3 不等长度
    for (let cutSize = minSize; cutSize <= 3; cutSize++) {
      const cut = s.slice(s.length - cutSize);
      if (cut && Number(cut) <= 255) {
        // 长度大于1的话，不能以0开头
        if (cut.length > 1 && cut[0] === "0") {
          // 为配合 continue 请使用 for 循环， while 循环不好使
          continue;
        }
        const tempArr1 = tempArr.concat();
        tempArr1.unshift(cut);
        backTracking(
          s.slice(0, s.length - cutSize),
          count + 1,
          minSize,
          tempArr1
        );
      }
    }
  }
  backTracking(s, 0, 1, []);
  return [...res];
}
// console.log(restoreIpAddresses("172162541"));

// 给定一个数组，找出三数之和为 0 的组合
function threeSum(nums) {
  // 回溯或循环找出所有三位组合，超时
}

// !先排序，选 i 为起点，剩下两个点从两端逼近
// 这个答案是我抄来的
function threeSumPro(nums = []) {
  res = [];
  nums.sort((a, b) => a - b); // 这里要小心，必须cb不然会出错
  const length = nums.length;
  for (let i = 0; i < length - 2; i++) {
    if (nums[i] > 0) break; // 起点都大于 0 了，放弃
    // 为了去重，起点选择有讲究
    if (i === 0 || nums[i] !== nums[i - 1]) {
      let low = i + 1,
        high = length - 1,
        sum = 0 - nums[i];
      // 两端逼近
      while (low < high) {
        if (nums[low] + nums[high] === sum) {
          res.push([nums[i], nums[low], nums[high]]);
          // 为了去重，low 和 high 连续移位
          while (low < high && nums[low] === nums[low + 1]) {
            low += 1;
          }
          while (high > low && nums[high] === nums[high - 1]) {
            high -= 1;
          }
          low += 1;
          high -= 1;
        } else if (nums[low] + nums[high] < sum) {
          low += 1;
        } else {
          high -= 1;
        }
      }
    }
  }
  return res;
}

// 一个二维矩阵，只有 0 和 1，求连着的 1 的最大个数
function maxAreaOfIsland(grid) {
  let res = 0;
  const hashmap = {};
  const maxRow = grid.length;
  const maxCol = grid[0].length;
  // 辅助函数，回溯和记忆
  function backTracking(i, j) {
    if (i < 0 || i === maxRow || j < 0 || j === maxCol || grid[i][j] === 0)
      return 0;
    const theKey = [i, j].toString();
    if (theKey in hashmap) return 0;
    hashmap[theKey] = true;
    return (
      1 +
      backTracking(i + 1, j) +
      backTracking(i - 1, j) +
      backTracking(i, j + 1) +
      backTracking(i, j - 1)
    );
  }
  // 遍历矩阵，遇到 1 尝试执行回溯
  for (let i = 0; i < maxRow; i++) {
    for (let j = 0; j < maxCol; j++) {
      if (grid[i][j] === 1) {
        res = Math.max(res, backTracking(i, j));
      }
    }
  }
  return res;
}

// 有序数组，某个位置旋转，变成两个有序序列，但中间有断崖
// 判断某个值是否存在，二分查找的变种
function searchRotated(nums, target) {
  // 先找出断崖波谷，然后根据与尾巴的比较，确定在断崖前还是后
  const n = nums.length;
  const bottomIdx = findBottom(nums);
  let low = target > nums[n - 1] ? 0 : bottomIdx;
  let high = target > nums[n - 1] ? bottomIdx : n - 1;
  while (low <= high) {
    const mid = low + Math.floor((high - low) / 2);
    const midValue = nums[mid];
    if (midValue === target) {
      return true;
    } else if (midValue < target) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return false;
}
// console.log(searchRotated([3, 1], 0));

// 找出断崖波谷
// 利用的是区间起点与 mid 值的比较，来不断缩小范围
// !注意终止条件
function findBottom(nums) {
  let n = nums.length,
    low = 0,
    high = n - 1;
  while (low <= high) {
    // ! tricky part，在两边逼近的过程中，全程都是 low > high，否则就说明找到了
    if (nums[low] <= nums[high]) {
      return low;
    }
    let mid = low + Math.floor((high - low) / 2);
    let midValue = nums[mid];
    if (midValue >= nums[low]) {
      low = mid + 1;
    } else {
      // 过程中有可能出现 mid 刚好是波谷，因此这里 high 不做进一步缩减
      high = mid;
    }
  }
}

// 最长连续递增区间
// 双指针 sliding window，当 j 遇到非递增数字或者到终点时，检查一下 tempMax 是否需要更新
function findLengthOfLCIS(nums) {
  let res = 0,
    i = 0,
    j = 0,
    n = nums.length;
  let prev = -Infinity;
  while (j < n) {
    if (nums[j] <= prev) {
      res = Math.max(res, j - i);
      i = j;
    }
    prev = nums[j];
    j += 1;
  }
  // 结束时再检查一下
  res = Math.max(res, j - i);
  return res;
}

// 找出第 K 个最大的元素
// 快排和它的双指针分区函数
// 注意第 K 个最大不是第 K 大，而是第 n - k 大
var findKthLargest = function(nums, k) {
  const n = nums.length;
  // 快排函数
  function quickSort(nums, left, right, k) {
    if (left > right) return; // 没有找到
    const p = partision(nums, left, right);
    if (p === n - k) {
      return nums[p];
    } else if (p < n - k) {
      return quickSort(nums, p + 1, right, k);
    } else {
      return quickSort(nums, left, p - 1, k);
    }
  }
  // 分区函数
  function partision(nums, left, right) {
    let i = left,
      j = left,
      target = nums[right]; // 原始写法，取区间最右作为比较值
    while (j <= right) {
      if (nums[j] <= target) {
        [nums[i], nums[j]] = [nums[j], nums[i]];
        i += 1;
      }
      j += 1;
    }
    // i - 1 就是分界点
    return i - 1;
  }
  // 执行快排判断
  return quickSort(nums, 0, nums.length - 1, k);
};

// 分区函数 pro，取随机下标作为比较
function partision(nums, left, right) {
  let i = left,
    j = left,
    ranNum = getRandomInt(left, right),
    target = nums[ranNum];
  while (j <= right) {
    if (nums[j] <= target) {
      [nums[i], nums[j]] = [nums[j], nums[i]];
      i += 1;
    }
    j += 1;
  }
  // 不是以最右作为对标的话，需要先找出对标元素当前的位置，将它与 i - 1交换
  const curIdx = nums.indexOf(target);
  [nums[i - 1], nums[curIdx]] = [nums[curIdx], nums[i - 1]];
  return i - 1;
}

// 随机取区间值，两边都包含
function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

// 最长连续序列，比如 [100, 4, 200, 1, 3, 2] 输出 4
// 思路：
// 如果内存是无限供应的，那显然可以用 200 个桶排序
// 折中一下，遍历数组构造哈希表 {1:true} 表示数字存在，同时遍历过程中记住数组的 min 和 max 值
// 从 min 到 max 之间，双指针 [i, j] sliding window，指针 j 每次自增 +1，检查是否存在哈希表里
// 这里从 min 开始每次自增 1 太慢了，如果数组方差很大，肯定超时
// 考虑到数组本身的长度肯定不大，我们应该遍历数组，从每个元素开始自增 + 1，因此 min 和 max 值也没有必要
// 遇到不存在的、以及到达终点后，更新一下 res 长度，即 j - i
function longestConsecutive(nums) {
  // !tricky 优化：直接转换为 set 作为哈希表，一举两得
  const hashmap = new Set(nums);
  let res = 0;
  // 遍历数组
  for (const num of nums) {
    // 优化：如果更小的数字存在，当前数字没有递增检查的意义
    if (hashmap.has(num - 1)) continue;
    let j = num;
    while (hashmap.has(j)) {
      j += 1;
    }
    res = Math.max(res, j - num);
  }
  return res;
}

// 第K个排列，给定数组 [1,2,3,...,n] 共有 n! 个排列，求其中第 K 大的排列
// 思路：
// 直觉告诉我们，先使用回溯列出所有排列，然后用快排找到目标
// 可优化点：快排分区的时候，选择随机参照物
function getPermutation(n, k) {
  // 列出所有排列
  const arr = [];
  // 辅助函数，回溯列出数组的排列
  function permutation(nums, temp) {
    if (!nums.length) {
      arr.push(Number(temp));
      return;
    }
    for (let i = 0; i < nums.length; i++) {
      if (temp.includes(nums[i].toString())) continue; // 千万别在这步 splice 数组，太丢人了
      permutation(nums, temp + nums[i]);
    }
  }
  const nums = [];
  for (let i = 1; i < n + 1; i++) {
    nums.push(i);
  }
  permutation(nums, "");
  // 对 arr 进行快排获取，这里仅做示例
  return findKthLargest(arr, k);
}

// 接上题，可优化的地方在于：跳跃式逼近 k
// 找出离 k 最近的数字 x 使得 x! < k < x+1!
// 每次找到 x 之后，根据 k / x! 可以知道在剩余的数字中，应该把哪个数字放首位
function getPermutationPro(n, k) {
  // 构造一个阶乘数组，方便快速查询
  const factorial = Array(n + 1).fill(null);
  let temp = 1,
    s = "";
  for (let i = 1; i < n + 1; i++) {
    temp *= i;
    s += i;
    factorial[i] = temp;
  }
  // TODO
  let res = "";
  function recur(s, k, guess) {
    if (k === 0) return;
    while (factorial[guess] > k) {
      guess -= 1;
    }
    // 找到 x，提取免排列前缀，同时找到剩余数字中，应该为首的那个
    res += s.slice(0, s.length - guess - 1);
    s = s.slice(s.length - guess - 1);
    const headIdx = Math.floor(k / factorial[guess]);
    // 重排剩余的 s
    s = s[headIdx] + s.slice(0, headIdx) + s.slice(headIdx + 1);
    k -= headIdx * factorial[guess];
    recur(s, k, guess);
  }
  recur(s, k, n);
  return res;
}
// console.log(getPermutationPro(9, 278893));

// 给定一组区间，合并区间
// 思路：
// 按起点大小排序，然后遍历比对 i[1] 和 i+1[0] 尝试合并
function merge(intervals) {
  interval.sort((a, b) => a[0] - b[0]);
  const res = [];
  let cur = intervals[0];
  for (let i = 1; i < intervals.length; i++) {
    const ele = intervals[i];
    if (cur[1] >= ele[0]) {
      if (cur[1] < ele[1]) cur[1] = ele[1];
    } else {
      res.push(cur);
      cur = ele;
    }
  }
  // 最后
  res.push(cur);
  return res;
}

// !给一个 n * n 矩阵，找出朋友圈个数，如果 1 和 2 是朋友，2 和 3 是朋友，那么1、2、3是一个朋友圈
function findCircleNum(M) {
  // DFS+哈希表记忆
  let res = 0;
  let n = M.length;
  const visited = Array(n).fill(false);
  // 总共就 n 个人，在朋友的圈子dfs都没见过的话，就是一个新的圈子，哪怕他自己就一个人
  for (let i = 0; i < n; i++) {
    if (!visited[i]) {
      res += 1;
      visited[i] = true;
      dfs(i); // 研究一下他的圈子
    }
  }
  function dfs(i) {
    // 填充 visited 数组，无需返回值
    for (let j = 0; j < n; j++) {
      // 这些条件是缩减的，所以会终止
      if (M[i][j] === 1 && !visited[j]) {
        visited[j] = true;
        dfs(j);
      }
    }
  }
  return res;
}

class Node {
  constructor(val) {
    this.val = val;
    this.next = null;
  }
}

// 原地合并有序链表
function mergeTwoLists(l1, l2) {
  // 不要把简单问题复杂化
  const dummy = new ListNode();
  let cur = dummy;
  while (l1 && l2) {
    if (l1.val <= l2.val) {
      cur.next = l1;
      l1 = l1.next;
    } else {
      cur.next = l2;
      l2 = l2.next;
    }
    cur = cur.next;
  }
  if (l1) cur.next = l1;
  if (l2) cur.next = l2;
  return dummy.next;
}

// TODO 递归写法
function mergeTwoListsByRecur(l1, l2) {}

// 单向链表排序
// 快排的分区函数需要交换节点，单向链表无法高效获得 prev 节点，因此优先考虑归并排序
function sortList(head) {
  if (!head.next) return head;
  if (!head) return null;
  let mid = findMiddle(head),
    right = sortList(mid.next);
  // ! mid.next 一定要置空
  mid.next = null;
  let left = sortList(head);
  return mergeTwoLists(left, right);
}

// 辅助函数，找出链表中间节点
function findMiddle(head) {
  let cur = head,
    fast = head;
  while (fast.next && fast.next.next) {
    fast = fast.next.next;
    cur = cur.next;
  }
  return cur;
}

// 环形链表II，找出环的起点
// 直觉告诉我们，使用哈希表是最简单的
function detectCycle(head) {
  // 因为 js 只允许 string 作为 object key，所以这里使用 set 更方便
  const hashmap = new Set();
  let cur = head;
  while (cur) {
    if (hashmap.has(cur)) {
      return cur;
    }
    hashmap.add(cur);
    cur = cur.next;
  }
  return null;
}

// 进阶解法，数学分析
// 先用快慢指针找到相遇的点 x
// 再用双指针分别从起点和 x 出发，如果再相遇，就是环的起点
function detectCyclePro(head) {}

// 二叉树的最近公共祖先，节点值唯一
// 思路：
// 由于节点无法向上查找，我们只能从 root 开始
// 如果有节点值与 root 相等，那就是 root
// 查找左树，如果找到 1 个节点，结果就是 root
// 如果没找到，往右树找
// 如果找到 2 个节点，继续以左树重复上述过程
// 我们每次 DFS 都尝试找两个节点

// 我们先看下平时 DFS 找节点的写法
function dfsBt(root, p) {
  if (!root) return false;
  return dfsBt(root.left) || root.val === p.val || dfsBt(root.right);
}

// 改造一下，每次尝试找两个节点
function dfsForTwo(root, p, q, score) {
  if (!root) return;
  if (root.val === p.val || root.val === q.val) score += 1;
  dfsForTwo(root.left, p, q, score);
  dfsForTwo(root.right, p, q, score);
}

// 再改进一下，有返回值的写法
// 输入 0，返回匹配的数量
// 树的 dfs 不会重复遍历，因此不会重复计算
function dfsForTwoScore(root, p, q) {
  if (!root) return 0;
  let cur = 0;
  if (root.val === p.val || root.val === q.val) cur += 1;
  return (
    cur + dfsForTwoScore(root.left, p, q) + dfsForTwoScore(root.right, p, q)
  );
}

// brute force 利用上述函数，来找公共父节点，缺点是遍历次数较多

// !这题其实有点难，特别是这种高效解法
// 关键是辅助函数，在递归返回的同时，适时操作外部变量
function lowestCommonAncestorPro(root, p, q) {
  let res = null;
  // 辅助函数，返回是否包含某节点
  function dfsForTwoScore(root, p, q) {
    if (!root) return 0;
    let cur = 0;
    if (root.val === p.val || root.val === q.val) cur = 1;
    const left = dfsForTwoScore(root.left, p, q);
    const right = dfsForTwoScore(root.right, p, q);
    if (left + cur + right === 2) res = root;
    return cur || left || right;
  }
  dfsForTwoScore(root, p, q);
  return res;
}

// 二叉树的按层交叉遍历，一正一反
var zigzagLevelOrder = function(root) {
  // bfs 的变种问题，需要维护中间变量 tempArr，以及一个布尔指针表示方向
  // 每一层的结束也需要一个标识
  if (!root) return [];
  const res = [];
  const q = [root, "#"];
  let swap = true;
  let tempArr = [];
  while (q.length) {
    const node = q.shift();
    if (node && node !== "#") {
      if (swap) {
        tempArr.push(node.val);
      } else {
        tempArr.unshift(node.val);
      }
    }
    // queue 的顺序不变，只是 tempArr 组装方式受 swap 影响
    if (node.left) q.push(node.left);
    if (node.right) q.push(node.right);

    // !检查是否该换行，并添加换行标识
    if (node === "#") {
      res.push(tempArr);
      tempArr = [];
      swap = !swap;
      if (q.length) q.push("#");
    }
  }
  return res;
};

// 实现一个 MinStack 使得可以 O(1) 获取最小值
// 存入对象，包含元素数值和当时的最小值，入栈出栈时都需要校正一下 this.min 指针
class MinStack {}

// LRU缓存机制
// 工业版是数组+哈希+双向链表+链表节点维护 hNext 支持哈希冲突解法
// 这里我们使用哈希表+双向链表简易实现
class BiNode {
  constructor(val, key) {
    this.val = val;
    this.key = key;
    this.prev = null;
    this.next = null;
  }
}
class LRUCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.hashmap = {};
    this.curSize = 0;
    this.biLinkedListHead = new BiNode(null);
    this.biLinkedListTail = new BiNode(null);
    this.biLinkedListHead.next = this.biLinkedListTail;
    this.biLinkedListTail.prev = this.biLinkedListHead;
  }
  put(key, value) {
    if (key in this.hashmap) {
      // 将节点移到尾巴，表示最近使用过
      this.moveNodeToTail(key);
      // 记得更新节点值
      this.hashmap[key].val = value;
    } else {
      if (this.curSize >= this.capacity) {
        // 满了，需要删除链表头部，就是最近最少使用的元素
        // 这里我们需要删除哈希表 key
        let node = this.biLinkedListHead.next;
        this.biLinkedListHead.next = node.next;
        node.next.prev = this.biLinkedListHead;
        this.hashmap[node.key] = null;
        this.curSize -= 1;
      }
      // 新建一个节点，放到尾巴，将哈希表 key 关联节点 node
      const newNode = new BiNode(value, key);
      newNode.next = this.biLinkedListTail;
      newNode.prev = this.biLinkedListTail.prev;
      this.biLinkedListTail.prev.next = newNode;
      this.biLinkedListTail.prev = newNode;
      this.hashmap[key] = newNode;
      this.curSize += 1;
    }
  }
  get(key) {
    if (!(key in this.hashmap) || !this.hashmap[key]) {
      return -1;
    }
    // 需要将目标节点放到链表尾巴，表示最近使用过
    this.moveNodeToTail(key);
    return this.hashmap[key].val;
  }
  // 公共能力，将节点放到尾巴
  moveNodeToTail(key) {
    const node = this.hashmap[key];
    // 将当前位置让出
    node.prev.next = node.next;
    node.next.prev = node.prev;
    // 连到尾巴上，这步操作有点骚
    node.next = this.biLinkedListTail;
    node.prev = this.biLinkedListTail.prev;
    this.biLinkedListTail.prev.next = node;
    this.biLinkedListTail.prev = node;
  }
}

// 接雨水，这题的解法正常人想不到，pass

// 反转链表，循环和递归两种解法
function reverseList(head) {
  let prev = null;
  while (head) {
    let next = head.next;
    head.next = prev;
    prev = head;
    head = next;
  }
  return prev;
}

// !递归，需要额外的 prev 参数，需要返回值
function reverseListRecur(head, prev) {
  if (!head) return prev;
  let next = head.next;
  head.next = prev;
  prev = head;
  head = next;
  return reverseListRecur(head, prev);
}

// 两个链表相加

function addTwoNumbers(l1, l2) {
  const dummy = new Node(null);
  let cur = dummy;
  let plusOne = 0; // 中间有可能是0 1 2
  while (l1 || l2) {
    const v1 = l1 ? l1.val : 0;
    const v2 = l2 ? l2.val : 0;
    const sum = v1 + v2 + plusOne;
    const num = sum % 10;
    plusOne = Math.floor(sum / 10);
    const node = new Node(num);
    cur.next = node;
    cur = cur.next;
  }
  // 最后需要检查一下 plusone
  if (plusOne > 0) {
    const node = new Node(1);
    cur.next = node;
  }
  return dummy.next;
}

// 买卖股票，允许多次交易，但一次只能买卖一笔
var maxProfit = function(prices) {
  let res = 0;
  for (let i = 1; i < prices.length; i++) {
    if (prices[i] > prices[i - 1]) {
      res += prices[i] - prices[i - 1];
    }
  }
  return res;
};

// 买卖股票，只允许最多一次交易
// 注意有前后顺序之分，尝试找出波峰波谷下标，然后错位匹配
function maxProfitOnce(prices) {
  let minArr = [];
  let maxArr = [];
  const n = prices.length;
  // 边界处理
  if (prices[0] < prices[1]) {
    minArr.push(0);
  }
  for (let i = 1; i < prices.length - 1; i++) {
    const price = prices[i];
    if (prices[i - 1] >= price && price < prices[i + 1]) {
      minArr.push(i);
    } else if (prices[i - 1] < price && price >= prices[i + 1]) {
      maxArr.push(i);
    }
  }
  if (prices[n - 1] > prices[n - 2]) {
    maxArr.push(n - 1);
  }
  // 从波峰波谷下标数组中（有序），错位匹配，事实上可以二分查找优化
  let res = 0;
  for (const i of minArr) {
    for (const j of maxArr) {
      if (j > i) {
        res = Math.max(res, prices[j] - prices[i]);
      }
    }
  }
}
