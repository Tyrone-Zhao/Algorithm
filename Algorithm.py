def BubbleSort(values):
    """ 冒泡排序 """
    for i in range(len(values)):
        flag = True
        for j in range(len(values) - i - 1):
            if values[j] > values[j + 1]:
                values[j], values[j + 1] = values[j + 1], values[j]
                flag = False

        if flag is True:
            break


def maxLongCharacters(string):
    """
        找出字符中最长连续的单个字符
        比如str = 'qqqeeedsdsdsfsdfsdfsdfsdfaaaaaaaaaaaaaaaaaaaaaaaaqqqqq'
        那么最长的连续单个字符就是: aaaaaaaaaaaaaaaaaaaaaaaa
    """
    matrix = [[""], [0]]
    for s in string:
        if s == matrix[0][-1]:
            matrix[1][-1] += 1
        else:
            matrix[0].append(s)
            matrix[1].append(1)

    max_num = max(matrix[1])
    i = matrix[1].index(max_num)

    return matrix[0][i], max_num


def quickSort(A, p, r):
    """
        快速排序，原址排序（原地修改）
        首次排序结果为：[p, i] <= base, [i+1, j-1] > base, [j, r - 1] 无限制，r为base。
        最好时间复杂度当q=n/2或（n-1）/2时为：nlgn, 最坏渐进下界为Omega n方
    :param A: 数组
    :param p: 0
    :param r: 数组长度减一
    :return: None
    """
    def partition(A, p, r):
        base = A[r]
        i = p - 1
        for j in range(p, r):
            if A[j] <= base:
                i += 1
                A[i], A[j] = A[j], A[i]
        A[i + 1], A[r] = A[r], A[i + 1]
        return i + 1

    if p < r:
        q = partition(A, p, r)
        quickSort(A, p, q - 1)
        quickSort(A, q + 1, r)


def circular_shift_left(int_value, k, bit=8):
    """
    循环左移
    :param int_value: 输入的整数
    :param k: 位移的位数
    :param bit: 整数对应二进制的位数
    :return: 循环移位后的整数值
    """
    bit_string = '{:0%db}' % bit
    bin_value = bit_string.format(int_value)  # 8 bit binary
    bin_value = bin_value[k:] + bin_value[:k]
    int_value = int(bin_value, 2)

    return int_value


def circular_shift_right(int_value, k, bit=8):
    """
    循环右移
    :param int_value: 输入的整数
    :param k: 位移的位数
    :param bit: 整数对应二进制的位数
    :return: 循环移位后的整数值
    """
    bit_string = '{:0%db}' % bit
    bin_value = bit_string.format(int_value)  # 8 bit binary
    bin_value = bin_value[-k:] + bin_value[:-k]
    int_value = int(bin_value, 2)

    return int_value


def bigLittleEndianConvert(data, need="big"):
    """ 大小端16进制数据转换 """
    import binascii
    import sys
    if sys.byteorder != need:
        return binascii.hexlify(binascii.unhexlify(data)[::-1])
    return data


def intToBin(X):
    """
    整数值转二进制
    :param X: 整数值
    :return: 32位二进制字符串
    """
    return '{:032b}'.format(X)[:32]


def loopLeftShift(a, k):
    res = list(intToBin(a))
    for i in range(k):
        temp = res.pop(0)
        res.append(temp)
    return int("".join(res), 2)



def binarySearch(nums, target):
    """ 二分查找 """
    i, j = 0, len(nums) - 1
    while i <= j:
        in_middle = (j + i) // 2
        if nums[in_middle] == target:
            return in_middle
        elif nums[in_middle] < target:
            i = in_middle + 1
        else:
            j = in_middle - 1

    return -1


def quickSort(myList, start, end):
    """ 快速排序 """
    # 判断low是否小于high,如果为false,直接返回
    if start < end:
        i, j = start, end
        # 设置基准数
        base = myList[i]

        while i < j:
            # 如果列表后边的数,比基准数大或相等,则前移一位直到有比基准数小的数出现
            while (i < j) and (myList[j] >= base):
                j -= 1

            # 如找到,则把第j个元素赋值给第个元素i,此时表中i,j元素相等
            myList[i] = myList[j]

            # 同样的方式比较前半区
            while (i < j) and (myList[i] <= base):
                i = i + 1
            myList[j] = myList[i]

        # 做完第一轮比较之后,列表被分成了两个半区,并且i=j,需要将这个数设置回base
        myList[i] = base

        # 递归前后半区
        quickSort(myList, start, i - 1)
        quickSort(myList, j + 1, end)


def distincList(nums):
    """ 列表去重 """
    nums.sort()
    i, j = 0, 1
    while i < len(nums) - 1:
        while j < len(nums):
            if nums[i] == nums[j]:
                del nums[j]
            else:
                j += 1
        i += 1
        j = i + 1

    return nums


def combination(n, m):
    """ 计算组合C(n, m) """
    return factorial(n) // factorial(m) // factorial(n - m)


def factorial(n):
    """ 计算阶乘 """
    c = 1
    for i in range(1, n + 1):
        c = c * i
    return c


def preTraverse(root):
    """
    前序遍历
    """
    if root is None:
        return
    print(root.value)
    preTraverse(root.left)
    preTraverse(root.right)


def midTraverse(root):
    """
    中序遍历
    """
    if root is None:
        return
    midTraverse(root.left)
    print(root.value)
    midTraverse(root.right)


def afterTraverse(root):
    """
    后序遍历
    """
    if root is None:
        return
    afterTraverse(root.left)
    afterTraverse(root.right)
    print(root.value)


def merge_sort(li):
    """
    归并排序
    :param li:
    :return:
    """
    # 不断递归调用自己一直到拆分成成单个元素的时候就返回这个元素，不再拆分了
    if len(li) <= 1:
        return li

    # 取拆分的中间位置
    mid = len(li) // 2
    # 拆分过后左右两侧子串
    left = li[:mid]
    right = li[mid:]

    # 对拆分过后的左右再拆分 一直到只有一个元素为止
    # 最后一次递归时候ll和lr都会接到一个元素的列表
    # 最后一次递归之前的ll和rl会接收到排好序的子序列
    ll = merge_sort(left)
    rl = merge_sort(right)

    # 我们对返回的两个拆分结果进行排序后合并再返回正确顺序的子列表
    # 这里我们调用拎一个函数帮助我们按顺序合并ll和lr
    return merge(ll, rl)


# 这里接收两个列表
def merge(left, right):
    # 从两个有顺序的列表里边依次取数据比较后放入result
    # 每次我们分别拿出两个列表中最小的数比较，把较小的放入result
    result = []
    while len(left) > 0 and len(right) > 0:
        # 为了保持稳定性，当遇到相等的时候优先把左侧的数放进结果列表，因为left本来也是大数列中比较靠左的
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    # while循环出来之后 说明其中一个数组没有数据了，我们把另一个数组添加到结果数组后面
    result += left
    result += right
    return result


class Solution:
    def toomCook3(self, num1, num2):
        """ 数字乘法 """
        import sympy as sy

        num1, num2 = int(num1), int(num2)
        # 数字分割
        base = 10000
        i = max(int(sy.log(num1, base)) // 3, int(sy.log(num2, base)) // 3) + 1
        B = base ** i

        d, m0 = divmod(num1, B)
        m2, m1 = divmod(d, B)

        d, n0 = divmod(num2, B)
        n2, n1 = divmod(d, B)

        # 评估
        po = m0 + m2
        p0 = m0
        p1 = po + m1
        p_1 = po - m1
        p_2 = (p_1 + m2) * 2 - m0
        pmax = m2

        qo = n0 + n2
        q0 = n0
        q1 = qo + n1
        q_1 = qo - n1
        q_2 = (q_1 + n2) * 2 - n0
        qmax = n2

        r0 = p0 * q0
        r1 = p1 * q1
        r_1 = p_1 * q_1
        r_2 = p_2 * q_2
        rmax = pmax * qmax

        # 插值求解
        a1 = [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1],
            [1, -2, 4, -8, 16],
            [0, 0, 0, 0, 1]
        ]

        a1 = sy.Matrix(a1)
        a1_I = a1.inv()  # 矩阵求逆

        a2 = sy.Matrix([r0, r1, r_1, r_2, rmax])
        ans = a1_I * a2

        return sum((ans[i, 0] * B ** i for i in range(len(ans))))


class Solution:
    from functools import lru_cache
    @lru_cache(100)
    def multiply(self, num1: str, num2: str) -> str:
        """
            字符串相乘, Karatsuba算法，n的1.58次方, https://zh.wikipedia.org/wiki/Karatsuba%E7%AE%97%E6%B3%95

            给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，
            它们的乘积也表示为字符串形式。

            输入: num1 = "123", num2 = "456"
            输出: "56088"
        """
        return str(self.karatsuba(num1, num2))

    def karatsuba(self, num1, num2):
        if not num1 or not num2:
            return 0
        n, m = len(num1), len(num2)
        if (n < 2) or (m < 2):
            return int(num1) * int(num2)

        maxLength = max(n, m)
        splitPosition = maxLength // 2
        high1, low1 = num1[:-splitPosition], num1[-splitPosition:]
        high2, low2 = num2[:-splitPosition], num2[-splitPosition:]
        z0 = self.karatsuba(low1, low2)
        z1 = self.karatsuba(str(int(low1) + int(high1 or "0")), str(int(low2) + int(high2 or "0")))
        z2 = self.karatsuba(high1, high2)

        return (z2 * 10 ** (2 * splitPosition)) + ((z1 - z2 - z0) * 10 ** (splitPosition)) + z0
