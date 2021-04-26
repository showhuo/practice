// 这个 fixed_point 是求解 f(x) = x 方程的经典方法
// 比如求解 x^^2 + 2x + 1 = 0，只需转换为 f(x) = -2 - 1/x，执行下列函数即可
// 当然了，并非所有方程都有解，这个可以通过限制递归次数或者提前数学证明
export function fixed_point(f, guess) {
  console.log(f(guess));
  return is_good_enough(f(guess), guess)
    ? guess
    : fixed_point(f, ((f(guess) + guess) / 2)); // 需要注意的是，很多方程并不能用这种方式逼近求解，特别是跳跃式的方程
}
function is_good_enough(a, b) {
  return Math.abs(a - b) < 0.001;
}
// console.log(fixed_point(x => 1 + 1 / x, 1)); // 黄金分割率

// 牛顿法提供了一种将任意 g(x) = 0 转换成 x = f(x) 的方法
// 有了 f(x) 我们就能用 fixed_point 求解 x = f(x) 也就是 g(x) = 0 的近似解
// 唯一需要做的就是，提供一个初步猜想的 guess 值
export function newtons_method(g, guess) {
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

// 因此牛顿法比简易的二分法准确得多，是严格有效的解法。