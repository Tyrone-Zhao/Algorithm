def twoSum(nums, target):
    """ 两数之和 """
    dict1 = {}
    for i in range(len(nums)):
        temp = target - nums[i]
        if dict1.get(temp) is not None:
            return [dict1[temp], i]
        else:
            dict1[nums[i]] = i


def lengthOfLongestSubstring(s):
    """ 无重复子串的长度 """
    st = {}
    i, ans = 0, 0
    for j in range(len(s)):
        if s[j] in st:
            i = st[s[j]] if st[s[j]] > i else i
        ans = ans if ans > j - i + 1 else j - i + 1
        st[s[j]] = j + 1
    return ans


def lengthOfLongestSubstring(s):
    """ 无重复子串的长度 """
    max_length, temp_length = 0, 0
    test = ''
    for i in s:
        if i not in test:
            test += i
            temp_length += 1
        else:
            if temp_length >= max_length:
                max_length = temp_length
            index = test.find(i)
            test = test[(index + 1):] + i
            temp_length = len(test)
    if temp_length > max_length:
        max_length = temp_length
    return max_length


def reverse(x):
    """ 整数反转 """
    if x == 0:
        return 0
    str_x = str(x)
    if str_x[0] == "-":
        rev = int("-" + str_x[len(str_x) - 1::-1].lstrip("0").rstrip("-"))
    else:
        rev = int(str_x[len(str_x) - 1::-1].lstrip("0"))
    if rev < -2 ** 31 or rev > 2 ** 31 - 1:
        return 0
    return rev


def reverse(x):
    """ 整数反转 """
    rev, temp = 0, 0
    max_v = 2 ** 31 - 1
    min_v = -2 ** 31

    while x != 0:
        if x < 0:
            pop = -(-x % 10)
            x = -(-x // 10)
        else:
            pop = x % 10
            x //= 10
        if rev > max_v // 10 or rev == max_v // 10 and pop > 7:
            return 0
        if rev < -(-min_v // 10) or rev == -(-min_v // 10) and pop < -8:
            return 0
        rev = rev * 10 + pop

    return rev


def longestPalindrome(s):
    """ 最长回文子串中心扩展解法 """
    if not s:
        return ""
    start, end = 0, 0
    for i in range(len(s)):
        len1 = expandAroundCenter(s, i, i)
        len2 = expandAroundCenter(s, i, i + 1)
        len_ = len1 if len1 > len2 else len2

        if len_ > end - start:
            start = i - (len_ - 1) // 2
            end = i + len_ // 2

    return s[start:end + 1]


def expandAroundCenter(s, left, right):
    """ 最长回文子串的中心扩展算法 """
    L, R = left, right
    while L >= 0 and R < len(s) and s[L] == s[R]:
        L -= 1
        R += 1

    return R - L - 1


def manacher(s):
    """ 最长回文子串马拉车算法 """
    if s == '':
        return ''
    t = '^#' + '#'.join(s) + '#$'
    c = d = 0
    p = [0] * len(t)
    for i in range(1, len(t) - 1):
        mirror = 2 * c - i
        p[i] = max(0, min(d - i, p[mirror]))

        while t[i + 1 + p[i]] == t[i - 1 - p[i]]:
            p[i] += 1
        if i + p[i] > d:
            c = i
            d = i + p[i]
    (k, i) = max((p[i], i) for i in range(1, len(t) - 1))
    return s[(i - k) // 2:(i + k) // 2]


def slidingWindow(s):
    """ 最长回文子串滑动窗口 """
    start, maxl = 0, 0
    for i in range(len(s)):
        if i - maxl >= 1 and s[i - maxl - 1:i + 1] == s[i - maxl - 1:i + 1][::-1]:
            start = i - maxl - 1
            maxl += 2
        if i - maxl >= 0 and s[i - maxl:i + 1] == s[i - maxl:i + 1][::-1]:
            start = i - maxl
            maxl += 1
    return s[start:start + maxl]


def findMedianSortedArrays(nums1, nums2):
    """ 寻找两个有序数组的中位数 """
    # 合并有序数组
    nums = nums1 + nums2

    # 排序有序数组
    nums.sort()

    # 求中位数
    remainder = len(nums) % 2
    half = len(nums) // 2
    if remainder:
        return nums[half]
    else:
        return (nums[half - 1] + nums[half]) / 2


def findMedianSortedArrays(nums1, nums2):
    """ 寻找两个有序数组的中位数 """
    m, n = len(nums1), len(nums2)
    if m > n:
        nums1, nums2 = nums2, nums1
        m, n = n, m
    i_min = 0
    i_max = m
    half_len = (m + n + 1) // 2
    while i_min <= i_max:
        i = (i_min + i_max) // 2
        j = half_len - i
        if i < i_max and nums2[j - 1] > nums1[i]:
            i_min = i + 1
        elif i > i_min and nums1[i - 1] > nums2[j]:
            i_max = i - 1
        else:
            if i == 0:
                max_left = nums2[j - 1]
            elif j == 0:
                max_left = nums1[i - 1]
            else:
                max_left = nums1[i - 1] if nums1[i - 1] > nums2[j - 1] else nums2[j - 1]
            if (m + n) % 2 == 1: return max_left

            if i == m:
                min_right = nums2[j]
            elif j == n:
                min_right = nums1[i]
            else:
                min_right = nums2[j] if nums2[j] < nums1[i] else nums1[i]

            return (max_left + min_right) / 2

    return 0.0


def romanToInt(s):
    """ 罗马数字转整数 """
    roman_numerals = {
        "M": 1000,
        "D": 500,
        "C": 100,
        "L": 50,
        "X": 10,
        "V": 5,
        "I": 1,
    }

    i = 0
    sum = 0
    while i < len(s) - 1:
        if roman_numerals[s[i]] < roman_numerals[s[i + 1]]:
            sum -= roman_numerals[s[i]]
        else:
            sum += roman_numerals[s[i]]
        i += 1

    return sum + roman_numerals[s[i]]


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def addTwoNumbers(l1: ListNode, l2: ListNode):
    """ 链表两数相加 """
    dummy_head = ListNode(0)
    p, q, curr, carry = l1, l2, dummy_head, 0
    while p or q:
        x = p.val if p else 0
        y = q.val if q else 0
        sum1 = carry + x + y
        carry = sum1 // 10
        curr.next = ListNode(sum1 % 10)
        curr = curr.next
        if p:
            p = p.next
        if q:
            q = q.next
    if carry > 0:
        curr.next = ListNode(carry)

    return dummy_head.next


def intToRoman(num):
    """ 整数转罗马数字 """
    if num < 1 or num > 3999:
        return None
    hashmap = {
        "M": 1000,
        "CM": 900,
        "D": 500,
        "CD": 400,
        "C": 100,
        "XC": 90,
        "L": 50,
        "XL": 40,
        "X": 10,
        "IX": 9,
        "V": 5,
        "IV": 4,
        "I": 1,
    }

    roman = ""
    while num:
        for k, v in hashmap.items():
            if num >= v:
                roman += k * (num // v)
                num -= v * (num // v)

    return roman


def threeSum(nums, target=0):
    """ 三数之和(target为0时双指针非最优解) """
    nums.sort()
    res = []
    for i in range(len(nums)):
        if i == 0 or nums[i] > nums[i - 1]:
            l = i + 1
            r = len(nums) - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s == target:
                    res.append([nums[i], nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                    while r > l and nums[r] == nums[r + 1]:
                        r -= 1
                elif s > target:
                    r -= 1
                else:
                    l += 1
    return res


def threeSum(nums):
    """ 三数之和为0 """
    dic = {}
    for num in nums:
        if num not in dic:
            dic[num] = 0
        dic[num] += 1
    if 0 in dic and dic[0] > 2:
        res = [[0, 0, 0]]
    else:
        res = []
    positive_num = (p for p in dic if p > 0)
    negative_num = (n for n in dic if n < 0)
    for p in positive_num:
        for n in negative_num:
            inverse = -(p + n)
            if inverse in dic:
                if inverse == p and dic[p] > 1:
                    res.append([p, n, inverse])
                elif inverse == n and dic[n] > 1:
                    res.append([p, n, inverse])
                elif inverse > p or inverse < n or inverse == 0:
                    res.append([p, n, inverse])
    return res


# 利用动态规划求解：
# 令dp[i][j]是s[0:i]和p[0:j]进行匹配，如果能匹配则为true，否则为false
# 动态规划状态方程式：
# 1 如果p[j] != '*'：
#  ]且(s[i] = p[j]或p[j] == '.')时，dp[i][j]为true，即
# dp[i][j] = i >= 1 and dp[i - 1][j - 1] and (s[i] == p[j] or p[j] == '.')
# 2 如果p[j] == '*':
# 1)p[j - 1]不用重复：
# dp[i][j] = dp[i][j - 2]
# 2)p[j - 1]至少重复一次：
# 当p[0:j]也能够匹配s[0:i - 1]，并且(s[i] = p[j - 1]或p[j - 2] = '.')时，dp[i][j]为true，即
# dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] and (s[i] == p[j - 1] or p[j - 1] == '.'))
def isMatch(s, p):
    """ 正则表达式.和*匹配 """
    # 如果是空串则为“#”，若都为空串也可以进行比较
    s = '#' + s
    p = '#' + p
    lengs = len(s)
    lengp = len(p)
    dp = [[False for i in range(lengp)] for j in range(lengs)]
    dp[0][0] = True
    # 如果p空，s不空，则不可能匹配；如果s空而p不空，还是有可能可以匹配的,所以s要从空串开始进行
    for i in range(0, lengs):
        for j in range(1, lengp):
            if p[j] != '*':  # 如果不是“*”，则直接看对应的是否匹配,i=0是初始状态
                dp[i][j] = i >= 1 and dp[i - 1][j - 1] and (s[i] == p[j] or p[j] == '.')
            else:
                # 如果是“*”，则判断是否重复0次还是重复多次（如果重复多次的话，还保证p[0:j]和s[0:i - 1]也能够匹配）
                # i=0的时候主要是在这里生效
                dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] and (s[i] == p[j - 1] or p[j - 1] == '.'))
    return dp[lengs - 1][lengp - 1]


def isValid(s):
    """ 有效的括号 """
    right = ""
    for i in s:
        if i == "(":
            right += ")"
        elif i == "[":
            right += "]"
        elif i == "{":
            right += "}"
        else:
            if i in [")", "]", "}"]:
                if not right or i != right[-1]:
                    return False
                else:
                    right = right[:-1]
    return not right


def isValid(s):
    """ 有效的括号 """
    hashmap = {
        ")": "(",
        "}": "{",
        "]": "[",
    }

    stack = []
    for i in range(len(s)):
        if s[i] in hashmap.keys():
            topElement = "#" if not stack else stack.pop()
            if topElement != hashmap[s[i]]:
                return False
        elif s[i] in hashmap.values():
            stack.append(s[i])
    return stack == []


def letterCombinations(digits):
    """ 电话号码的字母组合 """
    hashmap = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }
    list1 = []
    for i in range(len(digits) - 1, -1, -1):
        list1.append(hashmap[digits[i]])
    # 滑动窗口遍历values，每次取出两个输出结果
    if len(digits) == 1:
        return [x for x in list1[0]]
    elif not digits:
        return []
    else:
        # 取出头两个
        while list1:
            list2 = []
            list2.append(list1.pop())
            list2.append(list1.pop())
            result = []
            for i in range(len(list2) - 1, -1, -1):
                if i > 0:
                    j = i - 1
                    for k in list2[j]:
                        for l in list2[i]:
                            result.append(k + l)
                    j -= 1
            if list1:
                list1.append(result)
    return result


def letterCombinations(digits):
    """ 电话号码的字母组合 """
    if not digits:
        return []
    hashmap = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz",
    }
    result = ['']
    for d in digits:
        result = [a + b for a in result for b in hashmap[d]]
    return result


def fourSum(nums, target):
    """ 四数之和 """
    nums.sort()
    res = []
    for i in range(len(nums) - 3):
        if i > 0 and nums[i - 1] == nums[i]:
            continue
        for j in range(i + 1, len(nums) - 2):
            if j > i + 1 and nums[j - 1] == nums[j]:
                continue
            l = j + 1
            r = len(nums) - 1
            while l < r:
                four_sum = nums[i] + nums[j] + nums[l] + nums[r]
                if four_sum == target:
                    res.append([nums[i], nums[j], nums[l], nums[r]])
                    while l < r and nums[l + 1] == nums[l]:
                        l += 1
                    while l < r and nums[r - 1] == nums[r]:
                        r -= 1
                    l += 1
                    r -= 1
                elif four_sum < target:
                    l += 1
                else:
                    r -= 1
    return res


def getSum(nums, target, n):
    """ n数之和 """
    res = []
    if n < 2:
        return [[target]] if target in nums else []
    if n == 2:
        i = 0
        j = len(nums) - 1
        while i < j:
            tmp = nums[i] + nums[j]
            if tmp > target:
                j -= 1
            elif tmp < target:
                i += 1
            else:
                res.append([nums[i], nums[j]])
                while i < j and nums[i] == res[-1][0]:  # 注意
                    i += 1
    else:
        for i in range(len(nums) - n + 1):
            if (i and nums[i] == nums[i - 1]) or nums[i] + sum(nums[-n + 1:]) < target:
                continue
            if sum(nums[i:i + n]) > target:
                break
            tmp = getSum(nums[i + 1:], target - nums[i], n - 1)
            if tmp:
                for item in tmp:
                    res.append([nums[i]] + item)
    return res


def fourSum(nums, target):
    nums.sort()
    return getSum(nums, target, 4)


def generateParenthesis(n):
    """
        括号生成

        递归回溯遍历
        1. 选择。每一步一共有几种选择。本案例有两种选择 下一步添加"(" 或者 下一步添加")"
        2. 条件。每一种选择有什么约束。如果下一步添加"(", 剩余的左括号数必须大于0。如果下一步添加")", 剩余的右括号数必须大于左括号数。
        3. 结束。结束条件是什么。本案例的结束条件是左括号和右括号都添加完。得到一个结果。

        递归回溯：多于一个的递归才能回溯。本案例有两种条件的递归。所以当从某一种条件递归返回时，可能会触发下一个递归，此时将本层递归的状态继续传递下去，去探索从本层状态往后的其他可能性。

    """
    ans = []

    def backtrack(S='', left=0, right=0):
        if len(S) == 2 * n:
            ans.append(S)
            return
        if left < n:
            backtrack(S + '(', left + 1, right)
        if right < left:
            backtrack(S + ')', left, right + 1)

    backtrack()
    return ans


def divide(dividend, divisor):
    """
        两数相除, 用左移计算，左移等于乘以2

        * 解题思路：这题是除法，所以先普及下除法术语
        * 商，公式是：(被除数-余数)÷除数=商，记作：被除数÷除数=商...余数，是一种数学术语。
        * 在一个除法算式里，被除数、余数、除数和商的关系为：(被除数-余数)÷除数=商，记作：被除数÷除数=商...余数，
        * 进而推导得出：商×除数+余数=被除数。
        *
        * 要求商，我们首先想到的是减法，能被减多少次，那么商就为多少，但是明显减法的效率太低
        *
        * 那么我们可以用位移法，因为计算机在做位移时效率特别高，向左移1相当于乘以2，向右位移1相当于除以2
        *
        * 我们可以把一个dividend（被除数）先除以2^n，n最初为31，不断减小n去试探,当某个n满足dividend/2^n>=divisor时，
        *
        * 表示我们找到了一个足够大的数，这个数*divisor是不大于dividend的，所以我们就可以减去2^n个divisor，以此类推
        *
        * 我们可以以100/3为例
        *
        * 2^n是1，2，4，8...2^31这种数，当n为31时，这个数特别大，100/2^n是一个很小的数，肯定是小于3的，所以循环下来，
        *
        * 当n=5时，100/32=3, 刚好是大于等于3的，这时我们将100-32*3=4，也就是减去了32个3，接下来我们再处理4，同样手法可以再减去一个3
        *
        * 所以一共是减去了33个3，所以商就是33
        *
        * 这其中得处理一些特殊的数，比如divisor是不能为0的，Integer.MIN_VALUE和Integer.MAX_VALUE
        *
    """
    start = abs(divisor) << 31
    cur = abs(dividend)
    res = 0

    for i in range(32):
        y = cur - (start >> i)
        if y >= 0:
            cur = y
            res += (1 << (31 - i))

            if res >= (2 << 30) - 1:
                if (dividend > 0) != (divisor > 0):
                    return -(2 << 30)
                else:
                    return (2 << 30) - 1

    if (dividend > 0) != (divisor > 0):
        res = -res

    return res


def mergeKLists(lists):
    """
        合并K个排序链表

        res为头节点
        如果l.next不为none，则取值，放入temp
        然后排序temp
        创建链表, 返回res
    """
    a = []
    b = ListNode(0)
    c = b
    for i in lists:
        while i:
            a.append(i.val)
            i = i.next
    a.sort()
    for j in a:
        c.next = ListNode(j)
        c = c.next
    return (b.next)


def removeDuplicates(nums):
    """ 删除排序数组中的重复项 """
    if not nums:
        return 0
    i = 0
    j = i + 1
    while j <= len(nums) - 1:
        if nums[i] != nums[j]:
            nums[i + 1] = nums[j]
            i += 1
        j += 1
    return i + 1


def nextPermutation(nums):
    """
    下一个排列

    Do not return anything, modify nums in-place instead.

    因为是字典序排列，所以当一个排列不是完全逆序排列时，总可以将其重新排列使其字典序更大。 所以我们可以将一个排列分成左右两部分。
    右边的是一个完全逆序排列，左边剩余的部分称为前缀。 像这样：

    [a_1, a_2, ..., a_j-1] + [a_j, ..., a_n]

    根据前缀的长度可以分为两种情况，
    1、前缀为空列表 []
    2、前缀非空

    第一种情况，直接逆转整个列表即可。

    需要讨论的是第二种情况。

    首先注意到，若将前缀最后一个数添进右边部分，则右侧不再是完全逆序排列。因此可以对其进行重新排列使其字典序更大。根据题意，
    a_j-1只能被右侧大于它的最小数替换。替换后，新的右侧仍然是完全逆序排列。但是根据题意，新的右侧应该是升序排列，所以将其逆转即可。

    """
    n = len(nums)
    i = n - 1
    # = 号表示可以处理存在重复数字的情况
    while i and nums[i - 1] >= nums[i]:
        i -= 1
    if i:
        # 前缀不为空，nums[i]是右侧完全逆序排列的第一个数
        # nums[pre] 是前缀的最后一个数
        pre = i - 1
        j = i
        while j < n - 1 and nums[j + 1] > nums[pre]:
            j += 1
        # nums[j] 是右侧大于nums[pre] 的最小值
        nums[pre], nums[j] = nums[j], nums[pre]
    # 逆转右侧为升序排列
    j = n - 1
    while i < j:
        nums[i], nums[j] = nums[j], nums[i]
        i += 1
        j -= 1


class Solution:
    TOTAL = 0
    RUN = True

    def search(self, nums, target):
        """ 搜索旋转排序数组 """
        # 将数组一分为二，分别比头尾，尾大于头为有序，剩下的为无序
        i, j = 0, len(nums) - 1
        res = -1
        if nums and self.RUN:
            in_middle = (j + i) // 2
            list1 = nums[:in_middle + 1]
            list2 = nums[in_middle + 1:]
            if nums[in_middle] >= nums[i]:
                res = self.binarySearch(list1, target)
                if res == -1:
                    self.TOTAL += in_middle + 1
                    self.search(list2, target)
                else:
                    self.TOTAL += res
            else:
                res = self.binarySearch(list2, target)
                if res == -1:
                    self.search(list1, target)
                else:
                    self.TOTAL += in_middle + 1 + res

        if not self.RUN:
            return self.TOTAL
        return res

    def binarySearch(self, nums, target):
        """ 二分查找 """
        i, j = 0, len(nums) - 1
        while i <= j:
            in_middle = (j + i) // 2
            if nums[in_middle] == target:
                self.RUN = False
                return in_middle
            elif nums[in_middle] < target:
                i = in_middle + 1
            else:
                j = in_middle - 1

        return -1


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


def searchRange(nums, target):
    """
        在排序数组中查找元素的第一个和最后一个位置

        二分查找值，找到值后，用两个指针去找索引

    """
    i, j = 0, len(nums) - 1
    while nums and i <= j:
        in_middle = (i + j) // 2
        if nums[in_middle] == target:
            k, p = in_middle - 1, in_middle + 1
            while 0 <= k and nums[k] == nums[in_middle]:
                k -= 1
            while p <= len(nums) - 1 and nums[p] == nums[in_middle]:
                p += 1
            return [k + 1, p - 1]
        elif nums[in_middle] < target:
            i = in_middle + 1
        else:
            j = in_middle - 1
    return [-1, -1]


def isValidSudoku(board):
    """
        有效的数独

        用hashmap，先行列遍历，然后3*3遍历
    """
    rows = [{} for i in range(9)]
    columns = [{} for i in range(9)]
    boxes = [{} for i in range(9)]

    for i in range(9):
        for j in range(9):
            num = board[i][j]
            if num != '.':
                num = int(num)
                box_index = (i // 3) * 3 + j // 3

                rows[i][num] = rows[i].get(num, 0) + 1
                columns[j][num] = columns[j].get(num, 0) + 1
                boxes[box_index][num] = boxes[box_index].get(num, 0) + 1

                if rows[i][num] > 1 or columns[j][num] > 1 or boxes[box_index][num] > 1:
                    return False
    return True


def longestValidParentheses(s):
    """
        最长有效括号

        1.需有一个变量index记录有效括号子串的起始下标，res表示最长有效括号子串长度，初始值均为0
        2.用i作为下标遍历给字符串中的所有字符, 有两种情况, 遇到'('或者')' , 下面分别讨论这两种情况。
            2.1 遇到'(', 把'('的下标i入栈。
            2.2 遇到')', 有两种情况，栈为空，或者不为空。
                2.2.1 栈为空时，i之前的括号为无效括号，改变子串的起始位置，令index等于i+1。
                2.2.2 栈不为空时，因为只有两类括号，所以只要栈不为空，那么栈中的最后一个元素必然与右括号构成有效括号，pop出最后一个元素
                      此时栈分两种情况，为空或者不为空。
                      2.2.2.1 栈为空时，有效括号长度为当前下标i，减去有效括号的起始位置index并加1，比较有效括号长度和当前最大长度
                      2.2.2.2 栈不为空时，有效括号长度为当前下标i，减去栈中最后一个元素，比较有效括号长度和当前最大长度
    """
    res, index, stack = 0, 0, []  # 结果;
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)  # 记录索引
        elif not stack:
            index = i + 1  # 从下一位重新开始
        else:
            stack.pop()
            if stack:
                res = max(res, i - stack[-1])
            else:
                res = max(res, i - index + 1)
    return res


def longestValidParentheses(s):
    """
        最长有效括号


    """
    s = ')' + s
    dp = [0 for i in range(len(s))]
    for i, char in enumerate(s):
        if i == 0:
            continue
        if char == ')':  # 只有右括号时才可能有效
            if s[i - 1 - dp[i - 1]] == '(':  # 如果dp[i-1]范围前一个字符为左括号
                dp[i] = dp[i - 1] + 2
            dp[i] += dp[i - dp[i]]  # 加上dp[i]范围前一个字符的dp
    return max(dp)


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


def removeElement(nums, val):
    """
        移除元素，返回移除后的长度

        双指针一次遍历替换，算法复杂度O(n)
    """
    i = 0
    n = len(nums)
    while i < n:
        if nums[i] == val:
            nums[i] = nums[n - 1]
            n -= 1
        else:
            i += 1
    return n


def removeElement(nums, val):
    """
        移除元素，返回移除后的长度

        直接删除元素，算法复杂度O(n)
    """
    n = 0
    for i in range(len(nums)):
        if nums[i - n] == val:
            del nums[i - n]
            n += 1
    return len(nums)


def combinationSum(candidates, target):
    """
        组合总和

        1.排序
        2.target和c取余
    """
    candidates.sort()
    result = []
    n = len(candidates)
    for i in range(n):
        c = candidates[i]
        if i > 0 and c == candidates[i - 1]:
            continue
        if target % c == 0:
            result.append([c] * (target // c))
        for k in range(1, target // c):
            temp = [c] * (target // c - k)
            res = combinationSum(candidates[i + 1:], target % c + k * c)
            if res:
                for r in res:
                    result.append(temp + r)

    return result


class Solution:
    def __init__(self):
        self.res = []

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """
            组合总和, 不重复使用元素

            递归回溯，同时要去重。
            为啥能通过比较相同然后移动扫描下标就能去重？
            假设[i, j]区间中要找一个和为M。
            而如果[i+1, j]区间有好几个值跟v[i]相等，但是此区间==v[i]元素的个数一定比v[i, j]区间==v[i]元素的个数少；
            所以对于"含有v[i]的解"中，前者的答案一定包含后者，所以我们只需要求一次就行；
            后面相同的元素直接跳过去。

            图示：假设这个相同的数为3

            3 [3......3333.....3(第k个3)]4673435....
            i i+1                                   j

            既然区间[i+1, j]能够求出和为M的组合，其中包含了所有3的情况。
            而区间[i, j]3的个数比[i+1, j]3的个数更多，那么毫无疑问，[i, j]的解将覆盖[i+1, j]解中含有3的情况。
        """
        candidates.sort()  # 先排序
        self.getResult(candidates, target, [])  # []记录一个临时list,0记录位置保证不重复
        return self.res

    def getResult(self, nums, target, buff):
        if target == 0:
            self.res.append(list(buff))
        for i in range(len(nums)):
            if nums[i] > target or nums[i] in nums[:i]:
                continue
            buff.append(nums[i])
            self.getResult(nums[i + 1:], target - nums[i], buff)
            buff.pop()
        return


def combinationSum2(candidates, target):
    """ 组合总和, 不重复使用元素, 使用了n数之和函数 """
    candidates.sort()
    result = []
    for i in range(1, len(candidates) + 1):
        result += getSum(candidates, target, i)

    return result


class Solution:
    def __init__(self):
        self.res = []

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """
            组合总和 k个数的和为n

            递归回溯，如果len(buff) == k, self.res.append(list(buff))
        """
        nums = [x for x in range(1, 10)]
        self.backTracking(nums, n, [], k)

        return self.res

    def backTracking(self, nums, n, buff, k):
        if n == 0 and len(buff) == k:
            self.res.append(list(buff))
        for i in range(len(nums)):
            if nums[i] > n or nums[i] in nums[:i]:
                continue
            buff.append(nums[i])
            self.backTracking(nums[i + 1:], n - nums[i], buff, k)
            buff.pop()


class Solution(object):
    def solveSudoku(self, board):
        """ 解决9*9数独 """
        pos, H, V, G = [], [0] * 9, [0] * 9, [0] * 9  # 分别是：单元格的位置、水平位置、垂直位置、网格位置
        ctoV = {str(i): 1 << (i - 1) for i in range(1, 10)}  # 转换成1后面加i-1个0的形式，如:'4'=>1000
        self.vtoC = {1 << (i - 1): str(i) for i in range(1, 10)}  # 1后面加i-1个0的形式转换成数字，如:100=>'3'
        for i, row in enumerate(board):
            for j, c in enumerate(row):
                if c != '.':  # 如果是数独中的数字
                    v = ctoV[c]  # 获取1000式的值
                    # 计算行列和网格中出现的1~9的数字的次数，如1 | 100 | 1000，即为出现4, 3, 1
                    H[i], V[j], G[i // 3 * 3 + j // 3] = H[i] | v, V[j] | v, G[i // 3 * 3 + j // 3] | v
                else:
                    pos += (i, j),
        # 0*7 + 111111111 & (0000000000001010按位取反为1*8 + 11110101) = 0000000111110101，此为补码，即501。负数的补码最后一个1不动，其他位取反)
        # 计算未出现的值，如[393, 4], print(bin(393))可知[0b110001001, 3]，即数独中可填数字为9, 8, 4, 1
        posDict = {(i, j): [x, self.countOnes(x)] for i, j in pos \
                   for x in [0x1ff & ~(H[i] | V[j] | G[i // 3 * 3 + j // 3])]}
        print(posDict)
        self.slove(board, posDict)

    def countOnes(self, n):
        """
            计算n中1的数量
        """
        count = 0
        while n:
            count, n = count + 1, n & (n - 1)
        return count

    def slove(self, board, posDict):
        if len(posDict) == 0:
            return True
        p = min(posDict.keys(), key=lambda x: posDict[x][1])  # "."的位置，如(0, 2)
        candidate = posDict[p][0]
        while candidate:  # 如果某一个位置没有可填数字，那么就是无效数独
            v = candidate & (~candidate + 1)  # 获取最后一个1, 如0*8 + 10000100, 最后一个1为100, 值为4
            candidate &= ~v  # 从candidate中去掉最后一个1
            tmp = self.updata(board, posDict, p, v)  # 修改board和posDict
            if self.slove(board, posDict):  # 为下一个填值, 如果posDict为空，则代表都填完了，此时的数独为一个有效数独，那么结束递归
                return True
            self.recovery(board, posDict, p, v, tmp)  # 用recovery函数回溯
        return False

    def updata(self, board, posDict, p, v):
        """
            把最后一个1更新到board
            posDict中删除key为p的键值对
            posDict中的相关点去掉最后一个1，计数减1
            返回[posDict[p], 相关点的key]
        """
        i, j = p[0], p[1]
        board[i][j] = self.vtoC[v]  # 把100转为1~9的数字
        tmp = [posDict[p]]  # 如[[393, 4]]
        del posDict[p]
        for key in posDict.keys():
            if i == key[0] or j == key[1] or (i // 3, j // 3) == (key[0] // 3, key[1] // 3):  # 相关的点
                if posDict[key][0] & v:  # 如果可以填入相同元素，则需要去掉这个元素
                    posDict[key][0] &= ~v  # 去掉这个元素
                    posDict[key][1] -= 1  # 可选填数字减一
                    tmp += key,  # 把这些点记录下来，形如[[393, 4], (0, 4), (5, 4)]
        return tmp

    def recovery(self, board, posDict, p, v, tmp):
        """ 回溯恢复 """
        board[p[0]][p[1]] = '.'
        posDict[p] = tmp[0]  # 把这个点的位置和值添加回posDict
        for key in tmp[1:]:
            posDict[key][0] |= v  # 把最后一个1添加回相关的点中
            posDict[key][1] += 1  # 把相关点的计数加1


def strStr(haystack, needle):
    """
        实现strStr

        查找子串位置，sunday算法实现
    """
    hashmap = {}
    for i, c in enumerate(needle):
        hashmap[c] = i

    nn = len(needle)
    nh = len(haystack)
    i = 0
    while i <= nh - nn:
        j = nn - 1
        while j >= 0:
            if haystack[i + j] != needle[j]:
                if i + nn < nh:
                    skip = nn - hashmap.get(haystack[i + nn], -1)
                else:
                    return -1
                break
            j -= 1
        if j == -1:
            return i
        i += skip
    return -1


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        """ 全排列,给定一个没有重复数字的序列，返回其所有可能的全排列。 """
        res = []
        self.dfs(nums, [], res)
        return res

    def dfs(self, nums, path, res):
        if not nums:
            res.append(path)
        for i in range(len(nums)):
            self.dfs(nums[:i] + nums[i + 1:], path + [nums[i]], res)


class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """ 全排列II, 给定一个可包含重复数字的序列，返回所有不重复的全排列。"""
        res = []
        self.dfs(nums, [], res)
        return res

    def dfs(self, nums, path, res):
        if not nums:
            res.append(path)
        for i in range(len(nums)):
            if nums[i] in nums[:i]:
                continue
            else:
                self.dfs(nums[:i] + nums[i + 1:], path + [nums[i]], res)


def groupAnagrams(strs):
    """
        字母异位词分组

        输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
        输出:
        [
          ["ate","eat","tea"],
          ["nat","tan"],
          ["bat"]
        ]
    """
    chardict = {'a': 2, 'b': 3, 'c': 5, 'd': 7, 'e': 11, 'f': 13, 'g': 17, 'h': 19, 'i': 23, 'j': 29, 'k': 31, 'l': 37,
                'm': 41, 'n': 43, 'o': 47, 'p': 53, 'q': 59, 'r': 61, 's': 67, 't': 71, 'u': 73, 'v': 79, 'w': 83,
                'x': 89, 'y': 97, 'z': 101}
    ans = collections.defaultdict(list)

    for string in strs:
        temp = 1
        for char in string:
            temp *= chardict[char]
        # 用字符串内字符对应的质数值的乘积作为键值
        ans[temp].append(string)

    return list(ans.values())


def myPow(x, n):
    """
        Pow(x, n)
    """
    if n in (1, 0, -1): return x ** (n)
    return self.myPow(x, n // 2) ** 2 * (x if n % 2 != 0 else 1)


def firstMissingPositive(nums):
    """ 缺失的第一个正整数 """
    if not nums:
        return 1
    set1 = {i for i in range(1, len(nums) + 2)}
    set2 = set(nums)
    set1 -= set2

    return list(set1)[0]


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        """
            搜索插入位置

            给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引
            如果目标值不存在于数组中，返回它将被按顺序插入的位置，数组中无重复元素
        """
        i, j = 0, len(nums) - 1
        while i <= j:
            in_middle = (j + i) // 2
            if nums[in_middle] == target:
                return in_middle
            elif nums[in_middle] < target:
                if in_middle == len(nums) - 1 or target < nums[in_middle + 1]:
                    return in_middle + 1
                i = in_middle + 1
            else:
                j = in_middle - 1

        return 0


class Solution:
    def __init__(self):
        self.res = []

    def spiralOrder(self, matrix):
        """
            螺旋矩阵

            给定一个包含 m x n 个元素的矩阵（m 行, n 列），
            请按照顺时针螺旋顺序，返回矩阵中的所有元素。

            输入:
            [
             [ 1, 2, 3 ],
             [ 4, 5, 6 ],
             [ 7, 8, 9 ]
            ]
            输出: [1,2,3,6,9,8,7,4,5]
        """
        if not matrix:
            return self.res
        m, n = len(matrix), len(matrix[0])
        self.res += matrix.pop(0)
        if m < 2:
            return self.res
        elif m == 2:
            self.res += matrix.pop()[::-1]
            return self.res
        for i in range(0, m - 2):
            self.res.append(matrix[i].pop())
        self.res += matrix.pop()[::-1]
        if matrix[0]:
            for i in range(m - 3, -1, -1):
                self.res.append(matrix[i].pop(0))
        if matrix[0]:
            self.spiralOrder(matrix)
        return self.res


class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        """ 删除链表的倒数第N个节点 """

        def index(node):
            if not node:
                return 0
            i = index(node.next) + 1
            if i > n:
                node.next.val = node.val
            return i

        index(head)
        return head.next


class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        """ 删除链表的倒数第N个节点 """
        fast = slow = head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """ 合并区间 """
        if (len(intervals) <= 1):
            return intervals
        intervals.sort()
        rs = [intervals[0]]
        for i in range(1, len(intervals)):
            if (rs[-1][0] <= intervals[i][0] <= rs[-1][1]):
                rs[-1][1] = max(rs[-1][1], intervals[i][1])
            else:
                rs.append(intervals[i])
        return rs


class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        """
            螺旋矩阵2

            给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。
            输入: 3
            输出:
            [
             [ 1, 2, 3 ],
             [ 8, 9, 4 ],
             [ 7, 6, 5 ]
            ]
        """
        nums = (x for x in range(1, n ** n + 1))
        return self.spiralMatrix(nums, n)

    def spiralMatrix(self, nums, n):
        """ 生成螺旋矩阵"""
        res = [[] for i in range(n)]
        if not n:
            return res
        res[0] = [next(nums) for i in range(n)]
        if n == 1:
            return res
        if n > 2:
            temp1 = [[next(nums)] for i in range(1, n - 1)]

        res[-1] = [next(nums) for i in range(n)][::-1]
        if n > 2:
            temp2 = [[next(nums)] for i in range(1, n - 1)][::-1]
            ans = self.spiralMatrix(nums, n - 2)
            for i in range(0, n - 2):
                res[i + 1] = temp2[i] + ans[i] + temp1[i]

        return res


def generateMatrix(n):
    """
        螺旋矩阵2

        给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。
        输入: 3
        输出:
        [
         [ 1, 2, 3 ],
         [ 8, 9, 4 ],
         [ 7, 6, 5 ]
        ]
    """
    A = [[0] * n for _ in range(n)]
    i, j, di, dj = 0, 0, 0, 1
    for k in range(n * n):
        A[i][j] = k + 1
        if A[(i + di) % n][(j + dj) % n]:
            di, dj = dj, -di
        i += di
        j += dj
    return A


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


class Solution:
    def trap(self, height: List[int]) -> int:
        """
            接雨水

            给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
            输入: [0,1,0,2,1,0,1,3,2,1,2,1]
            输出: 6

            动态规划, 非最优解
        """
        if not height:
            return 0

        ans = 0
        size = len(height)
        left_max, right_max = [0] * size, [0] * size

        left_max[0] = height[0]
        for i in range(1, size):
            left_max[i] = max(height[i], left_max[i - 1])

        right_max[size - 1] = height[size - 1]
        for i in range(size - 2, -1, -1):
            right_max[i] = max(height[i], right_max[i + 1])

        for i in range(1, size - 1):
            ans += min(left_max[i], right_max[i]) - height[i]

        return ans


def trap(height):
    """
        接雨水

        给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
        输入: [0,1,0,2,1,0,1,3,2,1,2,1]
        输出: 6

        使用栈，非最优解
    """
    ans, current = 0, 0
    st = []
    while current < len(height):
        while st and height[current] > height[st[-1]]:
            top = st.pop()
            if not st:
                break
            distance = current - st[-1] - 1
            bounded_height = min(height[current], height[st[-1]]) - height[top]
            ans += distance * bounded_height

        st.append(current)
        current += 1
    return ans


def trap(height):
    """
        接雨水

        给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
        输入: [0,1,0,2,1,0,1,3,2,1,2,1]
        输出: 6

        动态规划
    """
    ans, left, right, left_max, right_max = 0, 0, len(height) - 1, 0, 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                ans += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                ans += right_max - height[right]
            right -= 1

    return ans


def trap(height):
    """
        接雨水

        给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
        输入: [0,1,0,2,1,0,1,3,2,1,2,1]
        输出: 6

        非最优解
    """
    if len(height) <= 1:
        return 0

    max_height = 0
    max_height_index = 0

    # 找到最高点
    for i in range(len(height)):
        h = height[i]
        if h > max_height:
            max_height = h
            max_height_index = i

    area = 0

    # 从左边往最高点遍历
    tmp = height[0]
    for i in range(max_height_index):
        if height[i] > tmp:
            tmp = height[i]
        else:
            area = area + (tmp - height[i])

    # 从右边往最高点遍历
    tmp = height[-1]
    for i in reversed(range(max_height_index + 1, len(height))):
        if height[i] > tmp:
            tmp = height[i]
        else:
            area = area + (tmp - height[i])

    return area


def countAndSay(n):
    """
        报数

        1.     1
        2.     11(表明1.有一个1)
        3.     21(表明2.有两个1)
        4.     1211(表明3.有一个2一个1)
        5.     111221(表明4.有一个1一个2两个1)
    """
    res = "1"
    if n <= 1:
        return res

    for i in range(1, n):
        ans = ""
        tuple_list = []
        for r in res:
            if tuple_list:
                if tuple_list[-1][0] == r:
                    tuple_list[-1][1] += 1
                else:
                    tuple_list.append([r, 1])
            else:
                tuple_list.append([r, 1])

        for t in tuple_list:
            ans += str(t[1]) + t[0]
        res = ans
    return res


def countAndSay(n):
    """
        报数

        1.     1
        2.     11(表明1.有一个1)
        3.     21(表明2.有两个1)
        4.     1211(表明3.有一个2一个1)
        5.     111221(表明4.有一个1一个2两个1)
    """
    s = '1'
    for _ in range(n - 1):
        s = re.sub(r'(.)\1*', lambda m: str(len(m.group(0))) + m.group(1), s)
    return s


def uniquePaths(m, n):
    """
        不同路径

        输入: m = 3, n = 2
        输出: 3
        解释:
        从左上角开始，总共有 3 条路径可以到达右下角。
        1. 向右 -> 向右 -> 向下
        2. 向右 -> 向下 -> 向右
        3. 向下 -> 向右 -> 向右
    """
    return combination(m + n - 2, n - 1)


def combination(n, m):
    """ 计算组合C(n, m) """
    return factorial(n) // factorial(m) // factorial(n - m)


def factorial(n):
    """ 计算阶乘 """
    c = 1
    for i in range(1, n + 1):
        c = c * i
    return c


def uniquePaths(m, n):
    """
        不同路径

        输入: m = 3, n = 2
        输出: 3
        解释:
        从左上角开始，总共有 3 条路径可以到达右下角。
        1. 向右 -> 向右 -> 向下
        2. 向右 -> 向下 -> 向右
        3. 向下 -> 向右 -> 向右
    """
    res = [[1] * m] + [[1] + [0] * (m - 1)] * (n - 1)
    for i in range(1, n):
        for j in range(1, m):
            res[i][j] = res[i - 1][j] + res[i][j - 1]

    return res[n - 1][m - 1]


def minPathSum(grid):
    """
        最小路径和

        输入:
        [
          [1,3,1],
          [1,5,1],
          [4,2,1]
        ]
        输出: 7
        解释: 因为路径 1→3→1→1→1 的总和最小。
    """
    if not grid:
        return 0
    r, c = len(grid[0]), len(grid)

    for i in range(1, r):
        grid[0][i] += grid[0][i - 1]
    for j in range(1, c):
        grid[j][0] += grid[j - 1][0]

    for i in range(1, r):
        for j in range(1, c):
            grid[j][i] += min(grid[j][i - 1], grid[j - 1][i])

    return grid[c - 1][r - 1]


def isValidBST(root):
    """
        验证二叉搜索树

        采用DFS中根遍历排序比较
    """
    stack, inorder = [], float('-inf')

    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        if root.val <= inorder:
            return False
        inorder = root.val
        root = root.right

    return True


def simplifyPath(path):
    """
        简化路径

        输入："/a//b////c/d//././/.."
        输出："/a/b/c"
    """
    s = path.split('/')
    res = []

    for x in s:
        if x:
            if x == '..':
                if res:
                    res.pop()
            elif x != '.':
                res.append(x)

    return '/' + '/'.join(res)


class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        """
            通配符匹配

            给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。
            s 可能为空，且只包含从 a-z 的小写字母。
            p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。
            输入:
            s = "aa"
            p = "a"
            输出: false
            解释: "a" 无法匹配 "aa" 整个字符串。
        """
        sn, pn = len(s), len(p)
        si = pi = 0
        save_si, save_pi = None, None
        while si < sn:
            if pi < pn and (p[pi] == '?' or p[pi] == s[si]):
                si += 1
                pi += 1
            elif pi < pn and p[pi] == '*':
                save_si, save_pi = si + 1, pi
                pi += 1
            elif save_pi is not None:
                si, pi = save_si, save_pi
            else:
                return False
        return p[pi:].count("*") == pn - pi


class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        """
            通配符匹配(非最优解）

            给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。
            s 可能为空，且只包含从 a-z 的小写字母。
            p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。
            输入:
            s = "aa"
            p = "a"
            输出: false
            解释: "a" 无法匹配 "aa" 整个字符串。

            动态规划

            dp[i][j]表示s到i位置,p到j位置是否匹配!

            初始化:

            dp[0][0]:什么都没有,所以为true
            第一行dp[0][j],换句话说,s为空,与p匹配,所以只要p开始为*才为true
            第一列dp[i][0],当然全部为False
            动态方程:

            如果(s[i] == p[j] || p[j] == "?") && dp[i-1][j-1] ,有dp[i][j] = true

            如果p[j] == "*" && (dp[i-1][j] = true || dp[i][j-1] = true)有dp[i][j] = true

            ​	note:

            ​	dp[i-1][j],表示*代表是空字符,例如ab,ab*

            ​	dp[i][j-1],表示*代表非空任何字符,例如abcd,ab*

            dp[i][j] incidates whether s[:i] matches p[:j].
            When we iterate p,

            If p[j] is not a *, we check whether s[:i] matches p[:j] and s[i] is c or ?. So dp[i+1][j+1] = dp[i][j] and (p[j] in {s[i], '?'})

            If p[j] is a *, we check whether s[:i] matches p[:j] and s[i+k:] matches p[j+1:] and k can be any value since a * can matches any length of string. Thus in this case, once we found a dp[i][j] is True(s[:i] matches p[:j]), we update entire dp[i:][j] to True enable any s[i+k] that matches p[j+1] to set dp[i+k][j+1] to True for next iteration.
            And we can generalize it as for i in range(len(s)): dp[i+1] |= dp[i].
            And if p[0] is *, we set dp[0] to True.

            And we can use dp rolling to reduce dp[i][j] to dp[i].

        """
        ls = len(s)
        if len(p) - p.count('*') > ls:
            return False
        dp = [True] + [False] * ls
        for c in p:
            if c == '*':
                for i in range(ls):
                    dp[i + 1] |= dp[i]
            else:
                for i in range(ls)[::-1]:
                    dp[i + 1] = dp[i] and (c in {s[i], '?'})
            dp[0] &= (c == '*')
        return dp[-1]


class Solution:
    @pysnooper.snoop()
    def maxSubArray(self, nums: list) -> int:
        """
            最大子序和

            给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

            输入: [-2,1,-3,4,-1,2,1,-5,4],
            输出: 6
            解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

            r[0]代表代表全局最优解
            r[1]以当前位置为结尾的局部最优解

            for i in range(1, len(nums)):
                nums[i] = max(nums[i], nums[i] + nums[i-1])
            return max(nums)
        """
        from functools import reduce
        return reduce(lambda r, x: (max(r[0], r[1] + x), max(r[1] + x, x)), nums, (max(nums), 0))[0]


class Solution:
    def maxSubArrayHelper(self, nums, l, r):
        if l > r:
            return -2147483647
        m = (l + r) // 2

        leftMax = sumNum = 0
        for i in range(m - 1, l - 1, -1):
            sumNum += nums[i]
            leftMax = max(leftMax, sumNum)

        rightMax = sumNum = 0
        for i in range(m + 1, r + 1):
            sumNum += nums[i]
            rightMax = max(rightMax, sumNum)

        leftAns = self.maxSubArrayHelper(nums, l, m - 1)
        rightAns = self.maxSubArrayHelper(nums, m + 1, r)

        return max(leftMax + nums[m] + rightMax, max(leftAns, rightAns))

    def maxSubArray(self, nums):
        """
            最大子序和（分治法）

            给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

            输入: [-2,1,-3,4,-1,2,1,-5,4],
            输出: 6
            解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

            r[0]代表代表全局最优解
            r[1]以当前位置为结尾的局部最优解

            for i in range(1, len(nums)):
                nums[i] = max(nums[i], nums[i] + nums[i-1])
            return max(nums)
        """
        return self.maxSubArrayHelper(nums, 0, len(nums) - 1)


class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        """
            字符串相乘（竖式计算，非最优解）

            给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，
            它们的乘积也表示为字符串形式。

            输入: num1 = "123", num2 = "456"
            输出: "56088"
        """
        ans = 0
        num1, num2 = num1[::-1], num2[::-1]
        s = [0] * (len(num1) + len(num2))
        for i in range(len(num1)):
            for j in range(len(num2)):
                s[i + j] = s[i + j] + int(num1[i]) * int(num2[j])
        for i in range(len(s)):
            ans += s[i] * 10 ** i
        return str(ans)


class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        """
            字符串相乘, 算法复杂度n的1.59次方

            给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，
            它们的乘积也表示为字符串形式。

            输入: num1 = "123", num2 = "456"
            输出: "56088"
        """
        if num1 == "0" or num2 == "0":
            return "0"
        n1, n2 = 0, 0
        t1, t2 = num1, num2 = int(num1), int(num2)
        while num1 >= 2:
            num1 >>= 1
            n1 += 1
        while num2 >= 2:
            num2 >>= 1
            n2 += 1
        r1, r2 = t1 % 2 ** n1, t2 % 2 ** n2
        A, B, C, D = 2 ** n1, r1, 2 ** n2, r2
        return str(A * C + B * C + A * D + B * D)


class Solution:
    def searchMatrix(self, matrix: list, target: int) -> bool:
        """
            搜索二维矩阵

            编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

            每行中的整数从左到右按升序排列。
            每行的第一个整数大于前一行的最后一个整数。

            输入:
            matrix = [
              [1,   3,  5,  7],
              [10, 11, 16, 20],
              [23, 30, 34, 50]
            ]
            target = 13
            输出: false
        """
        if not matrix or not matrix[0]:
            return False
        i, j = 0, len(matrix) - 1
        k = j
        while i <= j:
            in_middle = (j + i) // 2
            if matrix[in_middle][0] == target:
                return True
            elif matrix[in_middle][0] < target:
                if in_middle == k:
                    return self.binarySearch(matrix[in_middle], target)
                if matrix[in_middle + 1][0] > target:
                    return self.binarySearch(matrix[in_middle], target)
                i = in_middle + 1
            else:
                j = in_middle - 1
        return False

    def binarySearch(self, nums, target):
        """ 二分查找 """
        i, j = 0, len(nums) - 1
        while i <= j:
            in_middle = (j + i) // 2
            if nums[in_middle] == target:
                return True
            elif nums[in_middle] < target:
                i = in_middle + 1
            else:
                j = in_middle - 1

        return False


import time

start_time = time.time()
print(addTwoNumbers("MCMXCIV"))
end_time = time.time()
print(end_time - start_time)
