import numpy
import math

def isPalindrome(s):
  length = len(s)
  if length == 1:
    return True
  middle = math.floor(length/2)
  i = 0
  j = length - 1
  while i < middle and j >= middle:
    if s[i] != s[j]:
      return False
    i += 1
    j -= 1
  return True

# print(isPalindrome('abba'))

# 最长回文子串
# 解法一：从两边顶点找起，DP 转移方程为 P(i,j) = True if P(i+1,j-1) == True and S[i] == S[j] else False

def longestPalindrome(s):
  states = {}
  length = len(s)
  if length <= 1:
    return s

  def isFullfilled(i,j):
    if i == j:
      return True
    if j == i+1 and s[i] == s[j]:
      return True
    if j == i+1 and s[i] != s[j]:
      return False
    if 'k' + str(i) + 's' + str(j) in states:
      return states['k' + str(i) + 's' + str(j)]
    currentStatus = isFullfilled(i+1,j-1) == True and s[i] == s[j]
    states['k' + str(i) + 's' + str(j)] = currentStatus
    return currentStatus
  
  maxLen = -1
  result = ''
  for i in range(length):
    for j in range(length-1,i-1,-1):
      if isFullfilled(i,j) == True and j - i > maxLen:
        maxLen = j - i
        result = s[i:j+1]
  return result

test = "slvafhpfjpbqbpcuwxuexavyrtymfydcnvvbvdoitsvumbsvoayefsnusoqmlvatmfzgwlhxtkhdnlmqmyjztlytoxontggyytcezredlrrimcbkyzkrdeshpyyuolsasyyvxfjyjzqksyxtlenaujqcogpqmrbwqbiaweacvkcdxyecairvvhngzdaujypapbhctaoxnjmwhqdzsvpyixyrozyaldmcyizilrmmmvnjbyhlwvpqhnnbausoyoglvogmkrkzppvexiovlxtmustooahwviluumftwnzfbxxrvijjyfybvfnwpjjgdudnyjwoxavlyiarjydlkywmgjqeelrohrqjeflmdyzkqnbqnpaewjdfmdyoazlznzthiuorocncwjrocfpzvkcmxdopisxtatzcpquxyxrdptgxlhlrnwgvee"
# print(longestPalindrome(test))

# 解法二：循环遍历，找出每个元素为中心（或伪中心）时的最长回形子串，取其中最长的，也属于 DP 范畴

# 辅助函数，检查子串长度
def check(s,i,j):
  left = i
  right = j
  while left >=0 and right < len(s) and s[left] == s[right]:
      left -= 1
      right += 1
  return right - left - 1

def longestPalindromePlus(s):
  length = len(s)
  start = 0
  end = 0
  for i in range(length):
    # 检查最长子串
    len1 = check(s,i,i)
    len2 = check(s,i,i+1)
    maxLen = max(len1,len2)
    if maxLen > end - start:
      start = i - math.floor((maxLen-1)/2)
      end = i + math.floor(maxLen/2)
  return s[start:end+1]

print(longestPalindromePlus(test))

# 给定一个整数数组，求组成最大和的连续子串
# 思路：从左往右遍历一遍，当前和如果小于0则舍弃
def maxSubArray(nums):
  resultList = [nums[0]]
  currentSum = 0
  for i in nums:
    copyList = resultList.copy()
    if currentSum >= 0:
      currentSum += i
      copyList.append(i)
    else:
      currentSum = i
      copyList = [i]
    if sum(copyList) > sum(resultList):
      resultList = copyList.copy

