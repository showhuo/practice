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
    res += tailRecurWrong111(n - 1, res) + 2 * tailRecurWrong111(n - 2, res) + 3 * tailRecurWrong111(n - 3, res)
  }
  console.log(`n: ${n}, res: ${res}`);
  return res;
}
// tailRecurWrong111(15, 0);

// iterative function，尾巴递归
// 很难直接写出来，应该先写 loop + 变量交换，再翻译成递归
function f_loop_impl(n) {
  if (n < 3) return n;
  let count = 2, a = 2, b = 1, c = 0;
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
  return n < 3
    ? n
    : f_iterative_impl(2, 1, 0, n - 2);
}
function f_iterative_impl(a, b, c, count) {
  console.log(`a: ${a}, count: ${count}`);
  return count === 0
    ? a
    : f_iterative_impl(a + 2 * b + 3 * c, a, b, count - 1);
}
// f_loop_impl(15)
// f_iterative(15);


// 1.12 杨辉三角，第 n 行 m 列的元素，其中 m <= n，从 0 开始计数
function pascalTrianleEle(n, m) {
  if (n === 0 || m === 0 || m === n) return 1;
  if (n < 0 || m < 0) throw new Error('Invalid input');
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
  return calExponenTailRecur(b * b, n / 2, res)
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
  return a > b
    ? 1
    : term(a) * product(term, next(a), next, b);
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
    isOdd = !isOdd
    return isOdd
      ? a / (a + 1)
      : (a + 1) / a;
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
  return accumulateTail(multiplyTailRecur, 1, term, a, next, b)
}

// 1.35 常见的逼近方法，求解满足误差范围的解
// 这个 fixed_point 是求解 f(x) = x 方程的经典方法
// 比如求解 x^^2 + 2x + 1 = 0，只需转换为 f(x) = -2 - 1/x，执行下列函数即可
// 当然了，并非所有方程都有解，这个可以通过限制递归次数或者提前数学证明
function fixed_point(f, guess) {
  return is_good_enough(f(guess), guess)
    ? guess
    : fixed_point(f, (f(guess) + guess) / 2)
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
    return x => x - g(x) / derived(g)(x);
  }
  // 函数 g 的导数函数 g'
  function derived(g) {
    return x => (g(x + 0.00001) - g(x)) / 0.00001;
  }
  return fixed_point(newtons_transform(g), guess);
}

// 示例用牛顿法快速求解 x 的开方 sqrt(x)
function sqrtByNewton(x) {
  // 令 z 的平方等于 x，要求 z，因此将 g 表述为 z 的函数
  const g = z => Math.pow(z, 2) - x;
  return newtons_method(g, 1);
}
// console.log(sqrtByNewton(36));

// 1.40
function cubic(a, b, c) {
  return x => Math.pow(x, 3) + a * Math.pow(x, 2) + b * x + c;
}

// 1.41
function double(f) {
  return x => f(f(x));
}
function inc(x) {
  return x + 1;
}
// console.log(double(double(double))(inc)(5)); // 2^^4 * inc ?

// 1.42 compose 
function compose(...fns) {
  return num => fns.reduceRight((g, f) => f(g), num);
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
  return x => (f(x - dx) + f(x) + f(x + dx)) / 3;
}

// 1.46 iterative improvement: 更通用的方法论，从一个 good guess 开始，不断逼近答案。
function iterative_improvement(is_good_enough, improve) {
  return function iter(guess) {
    return is_good_enough(guess)
      ? guess
      : iter(improve(guess));
  }
}

function sqrt(n) {
  const is_good_enough = guess => Math.abs(guess * guess - n) < 0.001;
  const improve = guess => (guess + n / guess) / 2;
  return iterative_improvement(is_good_enough, improve)(1);
}
// console.log(sqrt(16));

function fixed_point(f) {
  const is_good_enough = guess => Math.abs(guess - f(guess)) < 0.0001;
  const improve = guess => (guess + f(guess)) / 2;
  return iterative_improvement(is_good_enough, improve)(1);
}