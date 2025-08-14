import random
from hashlib import sha256
from sympy import mod_inverse


p  = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
a  = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
b  = 0x28E9FA9E9D9F5E344D5AEF7E8D1B1055D0A9877CC62A474002DF32E52139F0A0
Gx = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
Gy = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
n  = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123


def inverse_mod(k, p):
    return mod_inverse(k, p)

def point_add(P, Q):
    if P is None:
        return Q
    if Q is None:
        return P
    if P[0] == Q[0] and P[1] != Q[1]:
        return None
    if P != Q:
        lam = ((Q[1] - P[1]) * inverse_mod(Q[0] - P[0], p)) % p
    else:
        lam = ((3 * P[0]**2 + a) * inverse_mod(2 * P[1], p)) % p
    x3 = (lam**2 - P[0] - Q[0]) % p
    y3 = (lam * (P[0] - x3) - P[1]) % p
    return (x3, y3)

def scalar_mult(k, P):
    R = None
    addend = P
    while k:
        if k & 1:
            R = point_add(R, addend)
        addend = point_add(addend, addend)
        k >>= 1
    return R


d = random.randint(1, n-1)
P = scalar_mult(d, (Gx, Gy))
print(f"[+] 私钥 d = {d}")
print(f"[+] 公钥 P = {P}")


def hash_msg(msg):
    h = sha256()
    h.update(msg.encode())
    return int(h.hexdigest(), 16)

def sign(msg, d, k=None):
    e = hash_msg(msg)
    while True:
        if k is None:
            k = random.randint(1, n-1)
        x1, y1 = scalar_mult(k, (Gx, Gy))
        r = (e + x1) % n
        if r == 0 or r + k == n:
            continue
        s = (inverse_mod(1 + d, n) * (k - r*d)) % n
        if s != 0:
            return (r, s, k)  # 返回 k 方便 PoC 漏洞
        k = None

def verify(msg, r, s, P):
    e = hash_msg(msg)
    t = (r + s) % n
    if t == 0:
        return False
    x1, y1 = point_add(scalar_mult(s, (Gx, Gy)), scalar_mult(t, P))
    R = (e + x1) % n
    return R == r


msg1 = "Hello"
msg2 = "World"

r1, s1, k_used = sign(msg1, d, k=12345)  # 强制重复 k
r2, s2, _ = sign(msg2, d, k=12345)

k_recovered = ((hash_msg(msg1) - hash_msg(msg2)) * mod_inverse(s1 - s2, n)) % n
d_recovered = ((s1 * k_recovered - hash_msg(msg1)) * mod_inverse(r1, n)) % n

print("\n[+] PoC: 重复 k 恢复私钥")
print(f"原私钥 d = {d}")
print(f"恢复私钥 d = {d_recovered}")


# 构造 r, s，使其满足验证
msg_fake = "Fake Message"
e_fake = hash_msg(msg_fake)
r_fake = random.randint(1, n-1)
s_fake = (r_fake * inverse_mod(d, n) - e_fake) % n

print("\n[+] 伪造签名")
print(f"伪造消息: {msg_fake}")
print(f"伪造签名 r = {r_fake}, s = {s_fake}")
print(f"验证伪造签名: {verify(msg_fake, r_fake, s_fake, P)}")
