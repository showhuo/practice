// 2.1-2 nonincreasing Insert-Sort
// 两层迭代，大迭代从第2位往右边走，小迭代从j-1往左边走，小迭代只跟 array[j] 比较大小
// 因为 array[0...j-1] 是已经排好序的，只要找到属于 array[j] 的位置，其他人右移一位
function nonIncreasingInsertSort(array) {
  const n = array.length;
  for (let j = 1; j < array.length; j++) {
    const element = array[j];
    let i = j;
    while (i > 0 && array[i-1] <= element ) {
      array[i] = array[i-1];
      i --;
    }
    // 小循环终止时，给 i 位赋值
    array[i] = element;
  }
  return array;
}

console.log(nonIncreasingInsertSort([1,2,2,3,4,5]));

// 2.1-3 循环不变式辅助证明线性搜索的正确性
// 三要素：循环开始前，条件成立
// 当次迭代前如果条件成立，迭代后条件肯定也成立
// 迭代结束时，有数据证明结论正确
// 此处的不变式可以是 if(v === array[index] && index < length) i = index
function searchIdx(array,v) {
  for (let index = 0; index < array.length; index++) {
    if(v === array[index]) return index;
  }
  return null;
}

// 2.1-4 两个数组相加
function add2Array(A,B) {
  const n = A.length = B.length;
  const res = Array(n+1).fill(null);
  let extra = 0;
  for (let index = n-1; index >= 0 ; index--) {
    const curSum = A[index] + B[index] + extra;
    const curNum = curSum % 10;
    // 如果 A B 长度不相等，断点处的进位就有可能是 2
    extra = Math.floor(curSum/10);
    res[index+1] = curNum;
  }
  res[0] = extra;
  return res;
}