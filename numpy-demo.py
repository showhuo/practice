import numpy as np

# 多维数组，shape 和 ndim 都是维度的标识
mula = np.array([[1,2,3,0],[4,4,6,1],[7,8,9,2]])
mula[1,1] = 5
# print(mula.shape)
# print(mula.ndim)
# print(mula)

# dtype 创造结构数组
personType = np.dtype({
    'names':['name','math','english','literature','total'],
    'formats':['S32','i','i','i','i']
})
students = np.array([('Lee',90,90,70,0),('Chen',80,80,80,0),('Wong',90,90,80,0)],dtype=personType)
names = students[:]['name']
maths = students[:]['math']
# print(names)
# print(np.mean(maths))

# 连续数组 算数运算
a = np.arange(1,11,2)
b = np.linspace(1,9,5)
# print('a is: %s' %a)
# print('b is: %s' %b)
# print(a+b)
# print(a*b)
# print(a%b)

# 求矩阵中最大值、最小值函数，示例二维数组，所以 axe 只有 0 和 1
mula = np.array([[1,5,3,0],[4,4,6,1],[3,8,9,2]])
amin = np.amin(mula)
amin0 = np.amin(mula,0)
amax1 = np.amax(mula,1)
amean = np.mean(mula)
amedian = np.median(mula)
# print('amin is %s \namin0 is %s \namax1 is %s' %(amin ,amin0 ,amax1))

# 40% 分界值
apercentile = np.percentile(mula,40)
# print(apercentile)

# 加权平均值、标准差、方差
a = np.array([1,2,3,4])
b = a.copy()
wta = np.average(a,weights=b)
std = np.std(a)
var = np.var(a)
# print('wta is %s \nstd is %s \nvar is %s' %(wta,std,var))

# 排序，可以自选方法，主要有 quicksort（默认）、mergesort、heapsort 三种
sortedA = np.sort(mula)
sortedAll = np.sort(mula,axis=None)
sortedAxe = np.sort(mula,axis=0)
# print('sortedA is \n%s \nsortedAll is \n%s \nsortedAxe is \n%s' %(sortedA,sortedAll,sortedAxe))

# 为求总分排序，更改结构数组，要求定义时提前预留位置
students[:]['total'] = students[:]['math'] + students[:]['english'] + students[:]['literature']
sortedStudents = np.sort(students,order='total')
print(students)
print(sortedStudents)
