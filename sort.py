# 插入排序
def InsertSort(arr=[]):
  length = len(arr)
  if length <= 1:
    return arr
  for i in range(1,length):
    value = arr[i]
    # !循环往前比较，如果比前方元素小，则将前方元素'后移'一位，'移动'的开销小于交换
    j = i - 1
    while j >= 0 and arr[j] > value:
      arr[j+1] = arr[j]
      j -= 1
    # 最后在目标位置替换插入 value
    arr[j+1] = value
  return arr
print(InsertSort([4,3,5,1,2,6]))