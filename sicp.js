// 动态替换函数
function minus(a, b) {
  return a - b;
}

function plus(a, b) {
  return a + b;
}

function a_plus_abs_b(a, b) {
  return (b > 0 ? plus : minus)(a, b);
}

// 求解平方根，恰好是尾巴递归
function sqrt(n) {
  function sqrtIter(guess) {
    console.log(`guess: ${guess}`);
    return goodEnough(guess) ? guess : sqrtIter(improve(guess));
  }
  function goodEnough(guess) {
    return Math.abs(guess * guess - n) < n * 0.01;
  }
  function improve(guess) {
    return (n / guess + guess) / 2;
  }
  return sqrtIter(1);
}
// sqrt(10);

// 普通递归与尾巴递归
// 1.11 f(n) = n if n < 3, f(n) = f(n-1)+2f(n-2)+3f(n-3) if n >= 3
function normalRecur111(n) {
  return n < 3
    ? n
    : normalRecur(n - 1) + 2 * normalRecur(n - 2) + 3 * normalRecur(n - 3);
}
// emm, this is not tail but normal recur...
function tailRecurWrong111(n, res) {
  if (n < 3) {
    res += n;
  } else {
    res +=
      tailRecurWrong111(n - 1, res) +
      2 * tailRecurWrong111(n - 2, res) +
      3 * tailRecurWrong111(n - 3, res);
  }
  console.log(`n: ${n}, res: ${res}`);
  return res;
}
// tailRecurWrong111(15, 0);

// iterative function，尾巴递归
// 很难直接写出来，应该先写 loop + 变量交换，再翻译成递归
function f_loop_impl(n) {
  if (n < 3) return n;
  let count = 2;
  let a = 2;
  let b = 1;
  let c = 0;
  while (count < n) {
    const d = c;
    c = b;
    b = a;
    a = a + 2 * c + 3 * d;
    count++;
    console.log(`a: ${a}, count: ${count}`);
  }
  return a;
}
// 尾递归版本
function f_iterative(n) {
  return n < 3 ? n : f_iterative_impl(2, 1, 0, n - 2);
}
function f_iterative_impl(a, b, c, count) {
  console.log(`a: ${a}, count: ${count}`);
  return count === 0 ? a : f_iterative_impl(a + 2 * b + 3 * c, a, b, count - 1);
}
// f_loop_impl(15)
// f_iterative(15);

// 1.12 杨辉三角，第 n 行 m 列的元素，其中 m <= n，从 0 开始计数
function pascalTrianleEle(n, m) {
  if (n === 0 || m === 0 || m === n) return 1;
  if (n < 0 || m < 0) throw new Error("Invalid input");
  return pascalTrianleEle(n - 1, m - 1) + pascalTrianleEle(n - 1, m);
}

// 1.16 求数字 b 的 n 次方，要求复杂度 lgn
// 普通递归
function calExponenRecur(b, n) {
  if (n === 0) return 1;
  return n % 2 === 1
    ? b * calExponenRecur(b, n - 1)
    : calExponenRecur(b * b, n / 2);
}
// 准备尾递归，先写出 loop + 参数迭代 + 过程累加
function calExponenLoop(b, n) {
  if (n === 0) return 1;
  let res = 1;
  while (n / 2 > 0) {
    if (n % 2 === 1) {
      n--;
      res *= b; // 过程累加
    }
    // !注意，这里不是 res*res，而是参数迭代
    b *= b;
    n /= 2;
    console.log(`n:${n}, res:${res}`);
  }

  return res;
}
// 尾递归的技巧：b * res 总是不变的，而 n 一直递减。关键就是找到不变的关系。
function calExponenTailRecur(b, n, res) {
  console.log(`n:${n}, b:${b}, res:${res}`);
  if (n === 0) return res;
  if (n % 2 === 1) {
    return calExponenTailRecur(b, n - 1, res * b);
  }
  return calExponenTailRecur(b * b, n / 2, res);
}
// console.log(calExponenRecur(2, 11));
// calExponenTailRecur(2, 11, 1);

// 1.18 用 add、double、halve 三个基础函数，拼装出 multiply 函数的尾递归版本，复杂度 lgn
function add(a, b) {
  return a + b;
}
function double(a) {
  return a * 2;
}
function halve(a) {
  return a / 2;
}

function multiplyTailRecur(a, b, temp = 0) {
  if (b === 0) return temp;
  if (b % 2 === 1) {
    return multiplyTailRecur(a, b - 1, add(temp, a));
  }
  // 这里的参数 a 和 temp 的迭代，很容易搞错，直觉可能认为 temp 应该翻倍，事实上不对，temp 只是用来暂存，核心还是 a 和 b 的倍数关系
  return multiplyTailRecur(double(a), b / 2, temp);
}
// console.log(multiplyTailRecur(11, 12, 0));

// 1.23 给定 next 函数，改造 smallest_divisor，使其每次猜两步，而不是一步
function next(n) {
  return n === 2 ? 3 : n + 2;
}
function smallest_divisor(n, a) {
  return n % a === 0 ? a : smallest_divisor(n, next(a));
}
// console.log(smallest_divisor(19999, 2));

// 1.30 累加范围值的递归函数，尾递归写法
function sumTailRecur(term, a, next, b) {
  function iter(a, res) {
    if (a > b) return res;
    return iter(next(a), res + term(a));
  }
  return iter(a, 0);
}

// 1.31 累乘范围值
function product(term, a, next, b) {
  return a > b ? 1 : term(a) * product(term, next(a), next, b);
}
function productTail(term, a, next, b) {
  function iter(a, res) {
    if (a > b) return res;
    return iter(next(a), res * term(a));
  }
  return iter(a, 1);
}
// 借助上述递归写出阶乘
function factorial(a, b) {
  function stay(a) {
    return a;
  }
  function next(a) {
    return a + 1;
  }
  return product(stay, a, next, b);
}
// console.log(factorial(2, 7));
function mockQuarterPai(n) {
  let isOdd = false;
  function term(a) {
    isOdd = !isOdd;
    return isOdd ? a / (a + 1) : (a + 1) / a;
  }
  function next(a) {
    return a + 1;
  }
  return productTail(term, 2, next, n);
}
// console.log(mockQuarterPai(1000) * 4);

// 1.32 更具普适性的 accumulate 函数，可以表示 sum 也可以表示 product
function accumulate(combiner, null_val, term, a, next, b) {
  return a > b
    ? null_val
    : combiner(term(a), accumulate(combiner, null_val, term, next(a), next, b));
}
function accumulateTail(combiner, null_val, term, a, next, b) {
  function iter(a, res) {
    if (a > b) return res;
    return iter(next(a), combiner(term(a), res));
  }
  return iter(a, null_val);
}
function sum(term, a, next, b) {
  return accumulate(add, 0, term, a, next, b);
}
function product(term, a, next, b) {
  return accumulateTail(multiplyTailRecur, 1, term, a, next, b);
}

// 1.35 常见的逼近方法，求解满足误差范围的解
// 这个 fixed_point 是求解 f(x) = x 方程的经典方法
// 比如求解 x^^2 + 2x + 1 = 0，只需转换为 f(x) = -2 - 1/x，执行下列函数即可
// 当然了，并非所有方程都有解，这个可以通过限制递归次数或者提前数学证明
function fixed_point(f, guess) {
  return is_good_enough(f(guess), guess)
    ? guess
    : fixed_point(f, (f(guess) + guess) / 2);
}
function is_good_enough(a, b) {
  return Math.abs(a - b) < 0.0001;
}
// console.log(fixed_point(x => 1 + 1 / x, 1)); // 黄金分割率
// console.log(fixed_point(x => -2 - 1 / x, -3));

// 牛顿法提供了一种将任意 g(x) = 0 转换成 x = f(x) 的方法
// 有了 f(x) 我们就能用 fixed_point 求解 x = f(x) 也就是 g(x) = 0 的近似解
// 唯一需要做的就是，提供一个初步猜想的 guess 值
function newtons_method(g, guess) {
  // 将 g 转换为 f
  function newtons_transform(g) {
    return (x) => x - g(x) / derived(g)(x);
  }
  // 函数 g 的导数函数 g'
  function derived(g) {
    return (x) => (g(x + 0.00001) - g(x)) / 0.00001;
  }
  return fixed_point(newtons_transform(g), guess);
}

// 示例用牛顿法快速求解 x 的开方 sqrt(x)
function sqrtByNewton(x) {
  // 令 z 的平方等于 x，要求 z，因此将 g 表述为 z 的函数
  const g = (z) => Math.pow(z, 2) - x;
  return newtons_method(g, 1);
}
// console.log(sqrtByNewton(36));

// 1.40
function cubic(a, b, c) {
  return (x) => Math.pow(x, 3) + a * Math.pow(x, 2) + b * x + c;
}

// 1.41
function double(f) {
  return (x) => f(f(x));
}
function inc(x) {
  return x + 1;
}
// console.log(double(double(double))(inc)(5)); // 2^^4 * inc ?

// 1.42 compose
function compose(...fns) {
  return (num) => fns.reduceRight((g, f) => f(g), num);
}
// console.log(compose(sqrtByNewton, inc)(8));

// 1.43 repeat(f,n)
function repeat(f, n) {
  return compose(...Array(n).fill(f));
}
// console.log(repeat(sqrtByNewton, 2)(5));

// 1.44 smoothed function: fs(x) = (f(x-dx)+f(x)+f(x+dx))/3
function smooth(f) {
  const dx = 0.00001;
  return (x) => (f(x - dx) + f(x) + f(x + dx)) / 3;
}

// 1.46 iterative improvement: 更通用的方法论，从一个 good guess 开始，不断逼近答案。
function iterative_improvement(is_good_enough, improve) {
  return function iter(guess) {
    return is_good_enough(guess) ? guess : iter(improve(guess));
  };
}

function sqrt(n) {
  const is_good_enough = (guess) => Math.abs(guess * guess - n) < 0.001;
  const improve = (guess) => (guess + n / guess) / 2;
  return iterative_improvement(is_good_enough, improve)(1);
}
// console.log(sqrt(16));

function fixed_point(f) {
  const is_good_enough = (guess) => Math.abs(guess - f(guess)) < 0.0001;
  const improve = (guess) => (guess + f(guess)) / 2;
  return iterative_improvement(is_good_enough, improve)(1);
}

// 以上是第一章的内容：通过函数构造抽象的概念。

// 第二章：通过数据构造抽象的概念。

// constructor 和 selector 是函数，它们是一组接口，连接数据的使用者和实现（可以有无数种实现，只要满足业务关系）
// 书中以有理数（也就是分数）作为示例，在代码中如何描述分数，而不是直接对它进行计算
// 先定义一种高级结构 pair，以及操作它的方法 head、tail
function pair(a, b) {
  // constructor 可以有很多种实现，数组、链表、对象、甚至函数，只需要给它配套的 selectors：head、tail，使其满足取第一位和第二位元素的能力即可。
  return [a, b];
}
function head(p) {
  return p[0];
}
function tail(p) {
  return p[1];
}
// 2.1 constructor 创建分数对象，有负数就给分子
function make_rat(n, d) {
  const g = gcd(n, d);
  return n * d > 0
    ? pair(Math.abs(n / g), Math.abs(d / g))
    : pair(-Math.abs(n / g), Math.abs(d / g));
}

// 2.2 用两个点表示一个片段，写出 constructor 和 selector
function make_segment(p1, p2) {
  return pair(p1, p2);
}
function start_segment(s) {
  return head(s);
}
function end_segment(s) {
  return tail(s);
}
// 用两个坐标表示一个点，写出 constructor 和 selector
function make_point(x, y) {
  return pair(x, y);
}
function x_point(p) {
  return head(p);
}
function y_point(p) {
  return tail(p);
}
// 用以上函数，表示一个片段的中点
function midpoint_segment(s) {
  const midX = x_point(start_segment(s));
  const midY = y_point(end_segment(s));
  return make_point(midX, midY);
}

// 2.3 基于 2.2 表示一个长方形，以及如何计算它的边长和面积
function make_rect(p1, p2, p4) {
  // 事实上还需要防御代码，对三个点进行检查，比如 p1-p2 与 y 轴平行，p1-p4 与 x 轴平行，不能重叠等等
  const sx = make_segment(p1, p4);
  const sy = make_segment(p1, p2);
  return pair(sx, sy);
}
function x_segment(r) {
  return head(r);
}
function y_segment(r) {
  return tail(r);
}
// 调用方
function perimeter(r) {
  return (x_segment(r) + y_segment(r)) * 2;
}
function area(r) {
  return x_segment(r) * y_segment(r);
}
// 改变一种 make_react 实现，要求调用方不用做修改
function make_react(p1, p3) {
  // 事实上还需要防御代码，p1 p3 必须是对角顶点，p1 在左下角
  const p2 = make_point(x_point(p1), y_point(p3));
  return pair(make_segment(p2, p3), make_segment(p1, p2));
}

// 2.4 可以用函数来表示 pair 数据结构，甚至可以说有三万种实现方式

function scope2_4() {
  function pair(x, y) {
    return (m) => (m === 0 ? x : y);
  }
  function head(p) {
    return p(0);
  }
  function tail(p) {
    return p(1);
  }
  // 这种函数表示要求 selector 传参也是函数！比上面要求传参普通数字更绕，但也预示了更强大的手法。
  function pair(x, y) {
    return (m) => m(x, y);
  }
  function tail(z) {
    return z((a, b) => b);
  }

  // 2.5
  function pair(a, b) {
    return Math.pow(2, a) * Math.pow(3, b);
  }
  function head(p) {
    // 要怎么从 pair 中取出 a 呢？不断整除 2 直到不能整除为止
    let a = 0;
    while (p % 2 === 0) {
      p /= 2;
      a += 1;
    }
    return a;
  }
}

// 2.12 电阻误差表示
function make_center_percent(c, p) {
  return make_interval(c * (1 - p), c * (1 + p));
}
function percent(i) {
  return (width(i) / center(i)) * 100;
}

function is_null(a) {
  return a === null || a === undefined;
}

// 用 pair 表示链表
// 注意！尾递归之前，需要想清楚 res 积累的方向：在 pair 这种定义里，显然从右往左积累比较简单
function list(...args) {
  function iter(index, res) {
    return index < 0 ? res : iter(index - 1, pair(args[index], res));
  }
  return iter(args.length - 1, null);
}
// console.log(list(1, 2, 3, 4).toString());

function listNormal(...args) {
  function recur(index) {
    return index === args.length ? null : pair(args[index], recur(index + 1));
  }
  return recur(0);
}
// console.log(listNormal(1, 2, 3));

// 2.17 last_pair 求 pair 链表最后一位
function last_pair(list) {
  return tail(list) === null ? head(list) : last_pair(tail(list));
}

// 2.18 翻转 pair 链表
// 为什么我感觉尾递归还更容易写一点。。。
function reverse(list) {
  function iter(cur, res) {
    return is_null(cur) ? res : iter(tail(cur), pair(head(cur), res));
  }
  return iter(list, null);
}

// pair 链表相连，教科书写法
// 思考一下，为什么这里的尾递归需要操作函数？
// 因为需要从右边开始积累，这是由 pair 这种结构决定的
function append(l1, l2) {
  return is_null(l1) ? l2 : pair(head(l1), append(tail(l1), l2));
}

// 2.19 找零钱的方法数量，请问硬币排列的顺序不同，是否影响答案？（不影响）
// 找零钱这种问题，因为答案难以积累，所以无法适用尾递归，传统递归加缓存才是最佳
function changes_coins(amount, coins) {
  return amount === 0
    ? 1
    : amount < 0
    ? 0
    : changes_coins(amount - head(coins), coins) +
      changes_coins(amount, tail(coins));
}

// 2.20 curry 函数
function brooks(curried, list) {
  function iter(list, res) {
    return is_null(list) ? res : iter(tail(list), res(head(list)));
  }
  return iter(list, curried);
}

function brooks_curried(list) {
  return brooks(head(list), tail(list));
}

// 经典的 map
function map(fun, items) {
  return is_null(items) ? null : pair(fun(head(items)), map(fun, tail(items)));
}
// console.log(map(x => x + 2, list(1, 2, 3, 4)));

// 2.21
function square(x) {
  return x * x;
}
function square_list(items) {
  return is_null(items)
    ? null
    : pair(square(head(items)), square_list(tail(items)));
}
function square_list(items) {
  return map(square, items);
}
// console.log(square_list(list(2, 3, 4)));

// 2.22 尾递归不好用的时候：因为链表是单向的，从左到右，我们没办法先累积右边的结果
// 那为什么翻转链表可以尾递归？因为翻转链表的尾递归是先积累左边
// 这里非要尾递归也行，获得结果后将链表拆开重组。。。
function square_list_tail(items) {
  function iter(items, res) {
    return is_null(items)
      ? res
      : iter(tail(items), pair(res, square(head(items))));
  }
  return iter(items, null);
}
// console.log(square_list_tail(list(2, 3, 4)));

// 2.23 不用返回值的递归写起来简单一些
function for_each(fun, items) {
  if (is_null(items)) return;
  fun(head(items));
  for_each(fun, tail(items));
}

// 2.27 deep_reverse
function deep_reverse(l) {
  return is_null(l)
    ? null
    : !is_pair(l)
    ? l
    : append(deep_reverse(tail(l)), deep_reverse(head(l)));
}
function is_pair(x) {
  return !is_null(head(x));
}
// console.log(deep_reverse(list(1, list(2, 3), 4)));

// 2.28 将嵌套 list 拍平为普通 list
function fringe(x) {
  return is_null(x)
    ? null
    : is_pair(x)
    ? append(fringe(head(x)), fringe(tail(x)))
    : list(x);
}
// console.log(fringe(list(list(1, 2), list(3, 4))));

// 2.29 一种叫 binary mobile 的数据
function make_mobile(l, r) {
  return list(l, r);
}
function left_branch(m) {
  return head(m);
}
function right_branch(m) {
  return tail(m);
}
function make_branch(length, structure) {
  return list(length, structure);
}
function branch_length(b) {
  return head(b);
}
function branch_structure(b) {
  return tail(b);
}

function total_weight(m) {
  if (!is_pair(m)) return m;
  const [l, r] = [left_branch(m), right_branch(m)];
  const [ls, rs] = [branch_structure(l), branch_structure(r)];
  return total_weight(ls) + total_weight(rs);
}

function balanced(m) {
  // 终点条件及其结果？很有意思，直接返回 true，可以解释为左右都为空，所以相等
  if (!is_pair(m)) return true;
  const [l, r] = [left_branch(m), right_branch(m)];
  const [ll, rl] = [branch_length(l), branch_length(r)];
  const [ls, rs] = [branch_structure(l), branch_structure(r)];
  const lrbanlance = ll * total_weight(ls) === rl * total_weight(rs);
  return !!(lrbanlance && balanced(ls) && balanced(rs));
}

// 2.30
function square_tree(l) {
  return is_number(l)
    ? square(l)
    : is_pair(l)
    ? pair(square_tree(head(l)), square_tree(tail(l)))
    : null;
}

function square_tree_map(l) {
  return map(
    (subTree) =>
      is_number(subTree)
        ? square(subTree)
        : is_pair(subTree)
        ? square_tree_map(subTree)
        : null,
    l
  );
}

// 2.31 组合结构的 map 需要判断节点类型，只有最基础的节点才能应用 fun 函数
// 有趣的是 tree_map 可以基于 map 进一步封装来实现
function tree_map(fun, tree) {
  return map(
    (subTree) =>
      is_null(subTree)
        ? null
        : !is_pair(subTree)
        ? fun(subTree)
        : tree_map(fun, subTree),
    tree
  );
}

// 不用 map 实现可能还更直观一点
function tree_map(fun, tree) {
  return is_null(tree)
    ? null
    : !is_pair(tree)
    ? fun(tree)
    : tree_map(fun, head(tree)) + tree_map(fun, tail(tree));
}
function times(a, b) {
  console.log(a * b);
  return a * b;
}
// 2.33 用 accumulate（也就是 reduce）来描述 map
// 这是正向的 reduce，fold_left
function accumulate_arr(fun, init, arr) {
  return is_null(arr)
    ? init
    : accumulate_arr(fun, fun(init, head(arr)), tail(arr));
}
// console.log(accumulate_arr(times, 1, list(1, 2, 4, 5)));
// 与传统 reduce 不同，这是 fold_right，不过 func 的参数顺序，积累仍然是正向的
// 可参考教科书习题 2.38
function accumulate_arr_reverse(func, init, arr) {
  return is_null(arr)
    ? init
    : func(head(arr), accumulate_arr_reverse(func, init, tail(arr)));
}
// 直觉并不容易想到，从左到右积累需要 func 自己负责
function map(f, sq) {
  return accumulate_arr_reverse((x, y) => pair(f(x), y), null, sq);
}

function append(sq1, sq2) {
  return accumulate_arr_reverse(pair, sq2, sq1);
}

function length(sq) {
  return accumulate_arr_reverse((x, y) => y + 1, 0, sq);
}

// 2.34
function horner_eval(x, co) {
  return accumulate_arr_reverse(
    (this_co, higher_co) => this_co + x * higher_co,
    0,
    co
  );
}
// console.log(horner_eval(2, list(1, 3, 0, 5, 0, 1)));

// 2.35
function count_leaves(t) {
  return accumulate_arr(
    (a, b) => a + b,
    0,
    map((subTree) => (is_pair(subTree) ? count_leaves(subTree) : 1), t)
  );
}

// 2.36
function accumulate_n(op, init, sq) {
  return is_null(head(sq))
    ? null
    : pair(
        accumulate_arr_reverse(
          op,
          init,
          map((subL) => head(subL), sq)
        ),
        accumulate_n(
          op,
          init,
          map((subL) => tail(subL), sq)
        )
      );
}

// 2.38 如果需要两种 reduce 即传统的 fold_left 和 fold_right 产生一样的结果
// 那么需要 op 满足交换律和结合律

// 2.39 用两种 reduce 分别实现 reverse，注意它们的区别，为了符合 pair list 的定义
// fold_right 指的是右边先积累完成，但仍然是左边 + 右边结合
// 两种 reduce 其实就是两种不同的递归方向/思路
function reverse(sq) {
  return fold_right((x, y) => append(y, list(x)), null, sq);
}

function reverse(sq) {
  return fold_left((x, y) => pair(y, x), null, sq);
}

// 2.40
function flatmap(f, items) {
  return accumulate(append, 0, map(f, items));
}
function get_intervals(n) {
  function iter(i, res) {
    return i > n ? res : iter(i + 1, append(res, list(i)));
  }
  return iter(1, list(1));
}
function unique_pairs(n) {
  return flatmap(
    (i) => map((j) => pair(i, j), get_intervals(i)),
    get_intervals(n)
  );
}
function prime_sum_pairs(n) {
  return map(make_pair_sum, filter(is_prime_sum, unique_pairs(n)));
}

// 2.41 找出有序的三元组，要求都小于 n，且和为 s
// 先列出所有三元组，再 filter
function orderedTriples(n, s) {}

// 2.42 八皇后
// 传统实现
function eightQueens() {
  const res = [];
  function iter(i, temp) {
    if (i === 8) {
      res.push(temp);
      return;
    }
    for (let j = 0; j < 8; j++) {
      if (is_safe(temp, i, j)) {
        // 需要注意的点，二维数组的深度复制
        const newTemp = JSON.parse(JSON.stringify(temp));
        newTemp[i][j] = true;
        iter(i + 1, newTemp);
      }
    }
  }
  // 辅助函数
  function is_safe(temp, i, j) {
    for (let k = 0; k <= i; k++) {
      if (temp[k][j]) return false;
    }
    let k = i;
    let l = j;
    while (k >= 0 && l >= 0) {
      if (temp[k][l]) return false;
      k--;
      l--;
    }
    (k = i), (l = j);
    while (k >= 0 && l <= 7) {
      if (temp[k][l]) return false;
      k--;
      l++;
    }
    return true;
  }
  const temp = Array(8).fill(new Array(8));
  iter(0, temp);
  return res;
}
console.log(eightQueens().length);

// FP 实现 TODO

// 2.45 高级操作 split
// 先思考应该返回什么，以及返回的对象（函数）会被如何使用
function split(op1, op2) {}

// 2.46 向量的一种实现
function make_vect(x, y) {
  return [x, y];
}
function x_vect(v) {
  return v[0];
}
function y_vect(v) {
  return v[1];
}

// 2.48 用两个起点相同的向量表示一条直线
// 对 constructor 来说，只依赖抽象，类似接口，不关心实现
function make_segment_by_vector(v1, v2) {
  return pair(v1, v2);
}
function start_segment_by_vector(s) {
  return head(s);
}
function end_segment_by_vector(s) {
  return tail(s);
}

// 2.54 两个 list 的比较
function equal(l1, l2) {
  return is_pair(l1)
    ? is_pair(l2)
      ? equal(head(l1), head(l2)) && equal(tail(l1), tail(l2))
      : false
    : is_pair(l2)
    ? false
    : l1 === l2;
}
// 可以更精简
function equal(l1, l2) {
  return is_pair(l1)
    ? is_pair(l2) && equal(head(l1), head(l2)) && equal(tail(l1), tail(l2))
    : l1 === l2;
}

// 2.57 改造 sum 和 product 组合，使其支持两个以上的传参
function make_sum_mul(l) {
  return reduceRight(make_sum, 0, l);
}

// 2.58 乘法优先于加法 TODO

// 2.59 Union set
function union_set(s1, s2) {
  return is_null(s1)
    ? s2
    : is_element_of_set(head(s1), s2)
    ? union_set(tail(s1), s2)
    : union_set(tail(s1), pair(head(s1), s2));
}

// 2.62 合并有序的集合，用新的集合 s3 最方便
function union_set_ordered(s1, s2) {}

// 2.68 Huffman encode 单个字符
// 因为是正向累积，优先尾递归
function encode_symbol(s, tree) {
  function iter(tree, res) {
    // 这里只能判断叶子节点，因为分支在下面判断过了，不能重叠
    return is_leaf(tree)
      ? res
      : is_member(s, left_branch(tree))
      ? iter(left_branch(tree), append(res, 0))
      : is_member(s, right_branch(tree))
      ? iter(right_branch(tree), append(res, 1))
      : null;
  }
  return iter(tree, list(null));
}

function encode_symbol(s, tree) {
  if (is_leaf(tree)) return null;
  if (is_member(s, left_branch(tree))) {
    return pair(0, encode_symbol(s, left_branch(tree)));
  }
  if (is_member(s, right_branch(tree))) {
    return pair(1, encode_symbol(s, right_branch(tree)));
  }
  return null;
}

// 2.69 将权重有序的叶子节点合成霍夫曼树
function successive_merge(orderedPairs) {
  function iter(orderedPairs) {
    if (!tail(orderedPairs)) return head(orderedPairs);
    const temp = make_code_tree(head(orderedPairs), head(tail(orderedPairs)));
    const left = addjoin_set(temp, tail(tail(orderedPairs)));
    return iter(left);
  }
  return iter(orderedPairs);
}

// 2.73 data-directed style
// 根据数据的 type，取出对应的函数来执行
function install_derive_sum() {
  function derive_sum(exp, variable) {
    return make_sum(deriv(addend(exp), variable), deriv(augend(exp), variable));
  }
  // 注册
  put("derive", "+", derive_sum);
}

function install_derive_product() {
  function derive_product(exp, variable) {
    return make_sum(
      make_product(multiplier(exp), deriv(multiplicand(exp), variable)),
      make_product(deriv(multiplier(exp), variable), multiplicand(exp))
    );
  }
  // 注册
  put("derive", "*", derive_product);
}

// 2.74 管理不同分公司的员工信息
function get_record(name, division) {
  return get("get_record", division.type)(name);
}

function get_salary(name, division) {
  return get("get_salary", division.type)(name);
}

function find_employ_record(name, divisions) {
  const targetDivision = filter((d) => is_member(name, d), divisions);
  return get("get_record", targetDivision.type)(name);
}

// 2.75 与数据驱动的风格不同，消息传递指的是，数据本身是函数，根据不同指令，返回不同结果
function make_from_mag_ang(m, a) {
  function dispatch(op) {
    return op === "mag"
      ? m
      : op === "ang"
      ? a
      : op === "real"
      ? m * math_cos(a)
      : op === "img"
      ? m * math_sin(a)
      : new Error("unknown op -- make_from_mag_ang");
  }
  return dispatch;
}

// Chapter 3

function make_withdraw(balance) {
  return (amount) => {
    if (balance >= amount) {
      balance -= amount;
      return balance;
    }
    return "Insufficient funds";
  };
}
// const w1 = make_withdraw(100);
// console.log(w1(30));
// console.log(w1(10));

// 3.1 用闭包 + local 变量积累状态，并提供操作接口
function make_accumulator(num) {
  return (x) => {
    num += x;
    return num;
  };
}

// 3.2 对函数调用进行内部计数，返回一个 dispatch 对象，支持多种操作（消息传递风格 + local 变量）
function make_monitored(f) {
  let count = 0;
  function dispatch(param) {
    if (param === "how many calls") {
      return count;
    }
    if (param === "reset count") {
      count = 0;
    } else {
      count++;
      return f(param);
    }
  }
  return dispatch;
}

// 3.3 给银行账户加密码，仍然是 dispatch + local 变量
function make_account(balance, pwd) {
  function withdraw(amount) {
    return balance >= amount
      ? (balance -= amount)
      : new Error("Insufficient funds");
  }
  function deposit(amount) {
    return (balance += amount);
  }
  function dispatch(enterPwd, action) {
    if (enterPwd !== pwd) {
      return () => new Error("Incorrect password");
    }
    switch (action) {
      case "withdraw":
        return withdraw;
      case "deposit":
        return deposit;
      default:
        return new Error("UnSupported action");
    }
  }
  return dispatch;
}

// 3.4 连续输错密码7次就报警。。。
function make_account_with_cops(balance, pwd) {
  // 其余跟上面一样
  let countBad = 0;
  function dispatch(enterPwd, action) {
    if (enterPwd !== pwd) {
      countBad++;
      if (countBad === 7) call_the_cops();
      return () => new Error("Incorrect password");
    }
    countBad = 0;
  }
  return dispatch;
}

// 3.5 借助 monte_carlo 函数（帮你计算给定的实验次数中，通过的比例）
function estimate_integral(P, x1, x2, y1, y2, trials) {
  const experiment = () => P(random_in_range(x1, x2), random_in_range(y1, y2));
  return monte_carlo(trials, experiment) * (x1 - x2) * (y1 - y2);
}

// 3.6 支持重现序列的（伪）随机函数
function rand(action) {
  let init = 0;
  function generate() {
    init = rand_update(init);
    return init;
  }
  function reset(val) {
    init = val;
    return init;
  }
  // 对外暴露 dispatch，非常灵活的高阶函数，既能返回值，也能返回函数
  function dispatch() {
    if (action === "generate") {
      return generate();
    }
    if (action === "reset") {
      return reset;
    }
  }
  return dispatch();
}

// 3.7 内部赋值提供了共同操作的可行性，比如银行子账户
function make_joint(account, originPwd, anotherPwd) {
  account(originPwd, "addSubAccount")(anotherPwd);
  return account;
}
// 需要增强原来的 make_account 使其允许开设子账户
function make_account(balance, pwd) {
  // 省略相同的
  function withdraw(amount) {
    balance -= amount;
    return balance;
  }
  function deposit(amount) {}
  const pwds = new Set([pwd]);
  function addSubAccount(pwd2) {
    pwds.add(pwd2);
  }
  function dispatch(enterPwd, action) {
    if (!pwds.has(enterPwd)) {
      return () => new Error("Incorrect password");
    }
    switch (action) {
      case "withdraw":
        return withdraw;
      case "deposit":
        return deposit;
      case "addSubAccount":
        return addSubAccount;
      default:
        return new Error("UnSupported action");
    }
  }
  return dispatch;
}

// 3.8 内部变量的存在，可能允许不同的执行顺序导致不同的结果
function g() {
  let fired = false;
  let cache;
  return (params) => {
    if (fired === false) {
      cache = params;
      fired = true;
    }
    return cache;
  };
}

// 3.17 pair 内部使用共享对象，那么 count 就需要特殊标记防止重复计算
function count_pair(x, cache) {
  return !is_pair(x) || cache.has(x)
    ? 0
    : count_pair(head(x), cache.add(x)) + count_pair(tail(x), cache.add(x)) + 1;
}

// 3.19 检测链表有环，要求空间复杂度 O(1)，需要用一个 tricky 办法：如果有环，slow 和 fast 指针一定会相遇
function listHasCycle(l) {
  let slow = l;
  let fast = l;
  while (fast) {
    slow = tail(slow);
    fast = tail(tail(fast));
    if (slow === fast) return true;
  }
  return false;
}

// 3.22 用函数来表示 queue
function make_queue() {
  let front_ptr = null;
  let end_ptr = null;
  function enqueue(ele) {
    if (!front_ptr) {
      front_ptr = pair(ele, null);
      end_ptr = front_ptr;
    } else {
      set_tail(end_ptr, pair(ele, null));
      end_ptr = tail(end_ptr);
    }
    return front_ptr;
  }
  // ...略
  return function dispatch(m) {
    if (m === "enqueue") return enqueue;
  };
}

// 3.47 信号量，允许有限的线程并发，假设 mutex 是系统提供的创建锁的 API
function make_semaphore(n) {
  const mutex = make_mutex();
  return (f) => {
    function semaphored_f(...arg) {
      if (n === 0) {
        mutex("lock");
      } else {
        n--;
      }

      const res = f.apply(...arg);
      n++;
      mutex("unLock");
      return res;
    }
    return semaphored_f;
  };
}

// 3.50 Stream 数据结构，其尾巴是函数，提供了懒加载懒执行的可能。
// Stream 是除变量赋值之外，FP 的另一种建模风格。

function stream_tail(s) {
  return tail(s)();
}
// 用 Stream 表示无限序列
function numFrom(n) {
  return pair(n, () => numFrom(n + 1));
}

console.log(numFrom(1));

// 在现实业务中，通常 memo 有返回值的函数，很少 memo 副作用
function memo(f) {
  let applied = false;
  let cache = null;
  return () => {
    if (!applied) {
      applied = true;
      cache = f();
    }
    return cache;
  };
}

// 3.55 输入 integer stream 输出 sum stream
function partial_sums(s) {
  function iter(s, sum) {
    return pair(head(s) + sum, () => iter(stream_tail(s), sum + head(s)));
  }
  return iter(s, 0);
}

// 3.64 stream_limit 只要序列相邻两个值相差在可接受范围内，就停止
function stream_limit(s, tolerance) {
  function iter(s) {
    const tailVal = head(stream_tail(s));
    return tailVal - head(s) <= tolerance ? tailVal : iter(stream_tail(s));
  }
  return iter(s);
}
