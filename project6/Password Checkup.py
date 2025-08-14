
import hashlib
import random
from typing import List, Tuple
from phe import paillier



N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551

def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def id_to_scalar(identifier: bytes, seed: bytes = b'') -> int:
    h = hashlib.sha256(seed + identifier).digest()
    s = int.from_bytes(h, 'big') % N
    return s if s != 0 else 1

class ScalarGroupElem:
    def __init__(self, s: int):
        self.s = s % N

    def __eq__(self, other):
        return isinstance(other, ScalarGroupElem) and self.s == other.s

    def __repr__(self):
        return f"Elem({hex(self.s)})"

    def exponentiate_by(self, exponent: int) -> "ScalarGroupElem":
        return ScalarGroupElem((self.s * (exponent % N)) % N)



class Party1:
    def __init__(self, ids: List[bytes], seed: bytes = b''):
        self.ids = ids
        self.seed = seed
        self.k1 = random.randrange(1, N)
        self.Z: List[ScalarGroupElem] = []

    def round1_send(self) -> List[ScalarGroupElem]:
        self.Z = [ScalarGroupElem(id_to_scalar(v, self.seed)).exponentiate_by(self.k1)
                  for v in self.ids]
        random.shuffle(self.Z)
        return self.Z

    def round3_receive_and_compute(self, received_pairs: List[Tuple[ScalarGroupElem, paillier.EncryptedNumber]],
                                   paillier_pubkey) -> Tuple[paillier.EncryptedNumber, int]:
        Zset = {z.s for z in self.Z}
        sum_cipher = None
        cnt = 0
        for g_elem, enc_t in received_pairs:
            gkk = g_elem.exponentiate_by(self.k1)
            if gkk.s in Zset:
                cnt += 1
                sum_cipher = enc_t if sum_cipher is None else sum_cipher + enc_t
        if sum_cipher is None:
            sum_cipher = paillier_pubkey.encrypt(0)
        # Refresh by adding encryption of 0
        refreshed = sum_cipher + paillier_pubkey.encrypt(0)
        return refreshed, cnt

class Party2:
    def __init__(self, pairs: List[Tuple[bytes, int]], seed: bytes = b''):
        self.pairs = pairs
        self.seed = seed
        self.k2 = random.randrange(1, N)
        self.paillier_pub = None
        self.paillier_priv = None

    def setup_paillier(self, keysize=1024):
        self.paillier_pub, self.paillier_priv = paillier.generate_paillier_keypair(n_length=keysize)
        return self.paillier_pub

    def round2_receive_Z_and_respond(self, Z_from_p1: List[ScalarGroupElem]) -> Tuple[List[ScalarGroupElem],
                                                                                       List[Tuple[ScalarGroupElem, paillier.EncryptedNumber]]]:
        Z2 = [elem.exponentiate_by(self.k2) for elem in Z_from_p1]
        random.shuffle(Z2)
        pairs = [(ScalarGroupElem(id_to_scalar(w, self.seed)).exponentiate_by(self.k2),
                  self.paillier_pub.encrypt(t)) for (w, t) in self.pairs]
        random.shuffle(pairs)
        return Z2, pairs

    def round3_receive_and_decrypt(self, enc_sum: paillier.EncryptedNumber) -> int:
        return self.paillier_priv.decrypt(enc_sum)

def demo():
    seed = b"session-seed-123"
    V = [b"userA", b"userB", b"userC", b"userD"]  # P1 IDs
    W_pairs = [(b"userX", 10), (b"userB", 7), (b"userC", 5), (b"userY", 3)]  # P2 IDs with values

    P1 = Party1(V, seed)
    P2 = Party2(W_pairs, seed)

    # Paillier key setup
    paillier_pub = P2.setup_paillier(keysize=1024)

    # Round 1
    msg1 = P1.round1_send()

    # Round 2
    Z2, pairs = P2.round2_receive_Z_and_respond(msg1)

    # Round 3
    refreshed_ct, intersection_size = P1.round3_receive_and_compute(pairs, paillier_pub)

    # Decrypt sum
    S = P2.round3_receive_and_decrypt(refreshed_ct)

    print("Intersection size (observed by P1):", intersection_size)
    print("Intersection sum (decrypted by P2):", S)
    expected = sum(t for (w, t) in W_pairs if w in V)
    print("Expected sum:", expected)

if __name__ == "__main__":
    demo()
