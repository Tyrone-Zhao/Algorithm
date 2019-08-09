package main

import (
	"bytes"
	"fmt"
	"strconv"
	"unsafe"
)

type SM3 struct {
	Input string
	IV    string
	V     []string
	B     []string
	W1    []uint32
	W2    []uint32
	A     []uint32
	MAX   uint32
}

// 初始化IV值
func (s *SM3) Init() {
	s.IV = "7380166f4914b2b9172442d7da8a0600a96f30bc163138aae38dee4db0fb0e4e"
	s.V = append(s.V, s.IV)
	s.MAX = 0xffffffff
}

// 输入字符补齐32位
func (s *SM3) CharToBin(char uint32) string {
	r := strconv.FormatUint(uint64(char), 2) // byte转为无符号整型
	// 	补充到32位
	if len(r) < 32 {
		r = fmt.Sprintf("%032s", r)
	}
	return r
}

// byte转8位二进制
func (s *SM3) ByteToBin(char byte) string {
	r := strconv.FormatUint(uint64(char), 2) // byte转为无符号整型
	// 	补充到32位
	if len(r) < 8 {
		r = fmt.Sprintf("%08s", r)
	}
	return r
}

// int转二进制32位
func (s *SM3) DecimalToBin(n int) string {
	return s.CharToBin(uint32(n))
}

// uint32转16进制
func (s *SM3) Uint32ToHex(i uint32) string {
	return fmt.Sprintf("%02x", i)
}

// 十六进制转二进制32位
func (s *SM3) HexToBin(x string) string {
	base, _ := strconv.ParseInt(x, 16, 33)
	return s.CharToBin(uint32(base))
}

// 十六进制转int
func (s *SM3) HexToInt(x string) int {
	var num int
	l := len(x)
	for i := l - 1; i >= 0; i-- {
		num += (int(x[l-i-1]) & 0xf) << uint8(i)
	}
	return num
}

// 8位十六进制转uint32
func (s *SM3) HexToUint32(x string) uint32 {
	base, _ := strconv.ParseInt(x, 16, 33)
	return uint32(base)
}

// 二进制转uint32
func (s *SM3) BinToUint32(b string) uint32 {
	base, _ := strconv.ParseInt(b, 2, 33)
	return uint32(base)
}

// 获取T值
func (s *SM3) GetT(j uint8) uint32 {
	var r uint32
	if j < 16 {
		base, _ := strconv.ParseInt("79cc4519", 16, 32)
		r = uint32(base)
	} else if j <= 63 {
		base, _ := strconv.ParseInt("7a879d8a", 16, 32)
		r = uint32(base)
	}
	return r
}

// 布尔函数FFj
func (s *SM3) FF(X uint32, Y uint32, Z uint32, j uint8) uint32 {
	var r uint32
	if j < 16 {
		r = X ^ Y ^ Z
	} else if j <= 63 {
		r = (X & Y) | (X & Z) | (Y & Z)
	}
	return r
}

// 布尔函数GGj
func (s *SM3) GG(X uint32, Y uint32, Z uint32, j uint8) uint32 {
	var r uint32
	if j < 16 {
		r = X ^ Y ^ Z
	} else if j <= 63 {
		r = (X & Y) | ((^X) & Z)
	}
	return r
}

// 循环左移, k代表左移的位数
func (s *SM3) LeftRotate(X uint32, k uint8) uint32 {
	X = (X << k) | (X >> (32 - k))
	return X
}

// 置换函数P0
func (s *SM3) P0(X uint32) uint32 {
	return X ^ s.LeftRotate(X, 9) ^ s.LeftRotate(X, 17)
}

// 置换函数P1
func (s *SM3) P1(X uint32) uint32 {
	return X ^ s.LeftRotate(X, 15) ^ s.LeftRotate(X, 23)
}

// 填充函数
func (s *SM3) FillInput() string {
	s.Init()
	var res string
	res += s.BigLittleEndianConvert()
	res += "1" // 将1添加到消息末尾
	temp := len(res) % 512
	if temp < 448 {
		res += s.SameString("0", 448 - temp) // (l + 1 + k) mod 512余448
	} else {
		res += s.SameString("0", 512 - temp + 448)
	}

	tail := fmt.Sprintf("%064s", s.DecimalToBin(len(s.Input)*8))
	res += tail
	return res
}

// 大小端数据转换, 实际上就是字节序反转
func (s *SM3) BigLittleEndianConvert() string {
	var temp string
	if s.IsLittleEndian() {
		for i := 0; i < len(s.Input); i++ { // 小端转大端
			temp += s.ByteToBin(s.Input[i])
		}
	} else {
		for i := len(s.Input) - 1; i >= 0; i-- {
			temp += s.ByteToBin(s.Input[i]) // 大端保持原状
		}
	}

	return temp
}

// 判断当前系统环境是否是小端
func (s *SM3) IsLittleEndian() bool {
	var i int32 = 0x01020304
	u := unsafe.Pointer(&i)
	pb := (*byte)(u)
	b := *pb
	return (b == 0x04)
}

// 返回固定长度相同字符的字符串
func (s *SM3) SameString(str string, n int) string {
	var buffer bytes.Buffer
	for i := 0; i < n; i++ {
		buffer.WriteString(str)
	}
	return buffer.String()
}

// 迭代压缩消息函数，返回64位16进制hash值
func (s *SM3) IterationCourse() string {
	str := s.FillInput()
	n := len(str) / 512
	for i := 0; i < n; i++ {
		s.B = append(s.B, str[i*512:(i+1)*512])
	}
	for i := 0; i < n; i++ {
		s.V = append(s.V, s.CF(s.V[i], s.B[i]))
		// 重置s.W1和s.W2
		s.W1 = []uint32{}
		s.W2 = []uint32{}
	}
	return s.V[len(s.V)-1:][0]
}

// 压缩函数
func (s *SM3) CF(Vi string, Bi string) string {
	// Vi按字存到A
	for i := 0; i < len(Vi)/8; i++ {
		s.A = append(s.A, s.HexToUint32(Vi[i*8:(i+1)*8]))
	}

	// 消息扩展，得到W1和W2
	s.informationExtend(Bi)

	for j := 0; j < 64; j++ {
		factor1 := s.LeftRotate(s.A[0], 12)
		factor2 := s.LeftRotate(s.GetT(uint8(j)), uint8(j%32))
		SS1 := s.LeftRotate((factor1 + s.A[4] + factor2), 7)
		factor3 := s.LeftRotate(s.A[0], 12)
		SS2 := SS1 ^ factor3
		TT1 := (s.FF(s.A[0], s.A[1], s.A[2], uint8(j)) + s.A[3] + SS2 + s.W2[j])
		TT2 := (s.GG(s.A[4], s.A[5], s.A[6], uint8(j)) + s.A[7] + SS1 + s.W1[j])
		s.A[3] = s.A[2]
		s.A[2] = s.LeftRotate(s.A[1], 9)
		s.A[1] = s.A[0]
		s.A[0] = TT1
		s.A[7] = s.A[6]
		s.A[6] = s.LeftRotate(s.A[5], 19)
		s.A[5] = s.A[4]
		s.A[4] = s.P0(TT2)
	}

	var res string
	for i := 0; i < len(Vi)/8; i++ {
		s.A[i] ^= s.HexToUint32(Vi[i*8 : (i+1)*8])
		res += s.Uint32ToHex(s.A[i])
	}
	return res
}

// 消息扩展函数
func (s *SM3) informationExtend(Bi string) {
	for i := 0; i < 16; i++ {
		s.W1 = append(s.W1, s.BinToUint32(Bi[i*32:(i+1)*32]))
	}
	for j := 16; j < 68; j++ {
		p := s.P1(s.W1[j-16] ^ s.W1[j-9] ^ s.LeftRotate(s.W1[j-3], 15))
		s.W1 = append(s.W1, p^s.LeftRotate(s.W1[j-13], 7)^s.W1[j-6])
	}

	for j := 0; j < 64; j++ {
		s.W2 = append(s.W2, s.W1[j]^s.W1[j+4])
	}
}

// 生成hash值
func (s *SM3) Hash(str string) string {
	s.Input = str
	s.Reset()
	return s.IterationCourse()
}

// 重置struct为零值
func (s *SM3) Reset() {
	s.V = []string{}
	s.B = []string{}
	s.A = []uint32{}
}

func main() {
	// 获取输入字符串
	var s SM3

	// "abc"测试
	fmt.Println(s.Hash("abc"))

	// 512bit测试
	fmt.Println(s.Hash("abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd"))
}
