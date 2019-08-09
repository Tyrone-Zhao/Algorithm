import binascii
import sys


class SM3:
    MAX = 2 ** 32
    IV = "7380166f4914b2b9172442d7da8a0600a96f30bc163138aae38dee4db0fb0e4e"

    def __init__(self, string):
        self.input = string
        self.b_input = bytes(self.input, "utf-8")
        self.hex_input = hex(int.from_bytes(self.b_input, "big"))[2:]  # 按照大端计算

    @property
    def hash(self):
        """
        获取结果Hash值
        :return: 256位16进制hash值
        """
        return self.iterationCourse(self.fill(self.hexToBin(self.hex_input)))[-1]

    def getT(self, j):
        """
        常量
        :param j:
        :return: 79cc4519 (0 <= j <= 15), 7a879d8a (16 <= j <= 63)
        """
        if 0 <= j <= 15:
            return int("79cc4519", 16)
        elif 16 <= j <= 63:
            return int("7a879d8a", 16)

    def FF(self, X, Y, Z, j):
        """
        布尔函数
        :param X: 16进制string
        :param Y: 16进制string
        :param Z: 16进制string
        :return: X ^ Y ^ Z (0 <= j <= 15), (X & Y) | (X & Z) | (Y & Z) (16 <= j <= 63)
        """
        if 0 <= j <= 15:
            return X ^ Y ^ Z
        elif 16 <= j <= 63:
            return (X & Y) | (X & Z) | (Y & Z)
        else:
            return 0

    def GG(self, X, Y, Z, j):
        """
        布尔函数
        :param X: 16进制string
        :param Y: 16进制string
        :param Z: 16进制string
        :return: X ^ Y ^ Z (0 <= j <= 15), (X & Y) | (～X & Z) (16 <= j <= 63)
        """
        if 0 <= j <= 15:
            return X ^ Y ^ Z
        elif 16 <= j <= 63:
            return (X & Y) | (self.non(X) & Z)
        else:
            return 0

    def P0(self, X):
        """
        置换函数
        :param X: 整数值
        :return: X ^ (X <<< 9) ^ (X <<< 17)
        """
        return X ^ self.leftRotate(X, 9) ^ self.leftRotate(X, 17)

    def P1(self, X):
        """
        置换函数
        :param X: 整数值
        :return: X ^ (X <<< 15) ^ (X <<< 23)
        """
        return X ^ self.leftRotate(X, 15) ^ self.leftRotate(X, 23)

    def leftRotate(self, X, k):
        """
        循环左移
        :param X: 整数值
        :param k: 位移的位数
        :param bit: 整数对应二进制的位数
        :return: 二进制左移k位的整数值
        """
        res = list(self.intToBin(X))
        for i in range(k):
            temp = res.pop(0)
            res.append(temp)
        return int("".join(res), 2)

    def fill(self, bin_string):
        """
        填充二进制消息string
        :param bin_string: 对任意消息使用self.msgToBin()得到的结果
        :return: 填充后的二进制消息m'，其比特长度为512的倍数
        """
        tail = "{:064b}".format(len(bin_string))
        bin_string += "1"
        div, mod = divmod(len(bin_string), 512)
        if mod >= 448:
            bin_string += "0" * (512 - mod + 448)
        else:
            bin_string += "0" * (448 - mod)

        return bin_string + tail

    def iterationCourse(self, msg):
        """
        迭代压缩消息
        :param msg: 填充后的二进制消息m', self.fill(msg)
        :return: Hash值(杂凑值)
        """
        # 将填充消息m'按512比特进行分组
        n = len(msg) // 512
        m = [msg[i * 512:(i + 1) * 512] for i in range(n)]
        # 对m[i]进行压缩迭代, msg = self.bigLittleEndianConvert(Vi, "big")  # 小端数据转换为大端
        V = [self.IV]
        for i in range(n):
            V.append(hex(int(self.intToBin(self.CF(V[-1], m[i]), 256), 2))[2:])

        return V

    def CF(self, Vi, mi):
        """
        压缩函数
        :param Vi: 512比特16进制数据
        :param mi: 512比特二进制数据
        :return: 512比特16进制数据
        """
        # 将Vi存储到字寄存器
        msg = Vi
        A = [int(msg[i * 8: (i + 1) * 8], 16) for i in range(len(msg) // 8)]

        # 消息扩展，得到W和W'
        W1, W2 = self.informationExtend(mi)

        # 压缩消息
        for j in range(64):
            factor1 = self.leftRotate(A[0], 12)
            factor2 = self.leftRotate(self.getT(j), j % 32)
            SS1 = self.leftRotate((factor1 + A[4] + factor2) % self.MAX, 7)
            factor3 = self.leftRotate(A[0], 12)
            SS2 = SS1 ^ factor3
            TT1 = (self.FF(A[0], A[1], A[2], j) + A[3] + SS2 + W2[j]) % self.MAX
            TT2 = (self.GG(A[4], A[5], A[6], j) + A[7] + SS1 + W1[j]) % self.MAX
            A[3] = A[2]
            A[2] = self.leftRotate(A[1], 9)
            A[1] = A[0]
            A[0] = TT1
            A[7] = A[6]
            A[6] = self.leftRotate(A[5], 19)
            A[5] = A[4]
            A[4] = self.P0(TT2)
        temp = self.intToBin(A[0], 32) + self.intToBin(A[1], 32) + self.intToBin(A[2], 32) + \
               self.intToBin(A[3], 32) + self.intToBin(A[4], 32) + self.intToBin(A[5], 32) + \
               self.intToBin(A[6], 32) + self.intToBin(A[7], 32)
        temp = int(temp, 2)
        return temp ^ int(Vi, 16)


    def informationExtend(self, mi):
        """
        消息扩展, 将512比特二进制消息扩展为132个字
        :param mi: Bi，512比特二进制数据
        :return: W -> 68个16进制消息字, W' -> 64个16进制消息字
        """
        # 第一步，将消息Bi划分为16个字W0~W15
        mi = self.binToHex(mi)
        W1 = [int(mi[i * 8: (i + 1) * 8], 16) for i in range(len(mi) // 8)]
        # 第二步
        for j in range(16, 68):
            p = self.P1(W1[j - 16] ^ W1[j - 9] ^ self.leftRotate(W1[j - 3], 15))
            W1.append(p ^ self.leftRotate(W1[j - 13], 7) ^ W1[j - 6])
        # 第三步
        W2 = [W1[j] ^ W1[j + 4] for j in range(64)]

        return W1, W2

    def bigLittleEndianConvert(self, data, need="big"):
        """
        大小端16进制数据转换
        :param data: 16进制小端string
        :param need: 转换为大端还是小端
        :return: 转换后16进制string
        """
        if sys.byteorder != need:
            return binascii.hexlify(binascii.unhexlify(data)[::-1])
        return data

    def hexToBin(self, hex_string):
        """
        十六进制string转换为二进制string
        :param hex: 十六进制string
        :return: 二进制string
        """
        res = ""
        for h in hex_string:
            res += '{:04b}'.format(int(h, 16))
        return res

    def binToHex(self, bin_string):
        """
        二进制string转换为十六进制string
        :param bin_string: 二进制string
        :return: 十六进制string
        """
        res = ""
        for i in range(len(bin_string) // 4):
            s = bin_string[i * 4: (i + 1) * 4]
            res += '{:x}'.format(int(s, 2))
        return res

    def msgToHex(self, msg):
        """
        字符串转换为16进制字符串
        :param msg: string
        :return: msg.encode("utf-8").hex()
        """
        return msg.encode("utf-8").hex()

    def msgToBin(self, msg):
        """
        字符串转换为二进制字符串
        :param msg: string
        :return: self.hexToBin(self.msgToHex(msg))
        """
        return self.hexToBin(self.msgToHex(msg))

    def intToBin(self, X, bits=32):
        """
        整数值转二进制
        :param X: 整数值
        :return: 32位二进制字符串
        """
        return ('{:0%db}' % bits).format(X)

    def non(self, X):
        """
        按位非
        :param X: 整数值
        :return: 按位非后的整数值
        """
        X = self.intToBin(X)
        Y = ""
        for i in X:
            if i == "0":
                Y += "1"
            else:
                Y += "0"
        return int(Y, 2)


if __name__ == "__main__":
    print(SM3("abc").hash)
    print(SM3("abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd").hash)
    print(SM3("你好").hash)



