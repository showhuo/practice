# 常见算法思想之一
# 要求本次的选择，不影响后面的选择

# 给定连续N天的股票价格，每次只能买卖/持有一手，求最大收益
# 等价于找波峰和波谷，求所有最长单调递增之和
def maxProfit(prices):
  bottom = prices[0]
  res = 0
  for i in range(1,len(prices)-1):
    p = prices[i]
    if p > prices[i-1] and p >= prices[i+1]:
      res += p - bottom
    elif p <= prices[i-1] and p <= prices[i+1]:
      bottom = p
  # 最后一位
  if prices[-1] > prices[-2]:
    res += prices[-1] - bottom
  return res

print(maxProfit([1,2,3,4,5]))