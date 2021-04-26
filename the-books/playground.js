// 渐进逼近或者暴力迭代，求 100*n*n < 2^^n 的转折点，谁更快？
import { fixed_point, newtons_method } from "./sicp/sicp-how-to-calculate-math-function.js";
// 借助 sicp 里对数学方程的近似求解思路
// 需要注意的是这种跳跃式的方程，只有牛顿解法是可靠的
const fooForNewton = n => 100 * n * n - Math.pow(2, n);
const fooForFixPoint = n => 100 * n - (Math.pow(2, n) / n);
const res2 = newtons_method(fooForNewton, 1);
console.log(res2);

// 经典问题：买卖一次股票，使得收益最大
// 最佳解法：遍历一次，记录 minPrice 和 maxProfit，取后者为答案
// 最佳解法二：转换为等价问题，最大（和）连续子序列，指的是股价变化值组成的序列
// 遍历一次，记录 maxCurSum 和 maxProfit，其中 maxCurSum 小于 0 时，重置为 0
function buyStockOnce(prices) {
    let minPrice = Infinity;
    let maxProfit = 0;
    for (const p of prices) {
        if (p < minPrice) minPrice = p;
        maxProfit = Math.max(maxProfit, p - minPrice);
    }
    return maxProfit;
}