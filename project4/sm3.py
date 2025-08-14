import hashlib
from typing import List, Optional

# Constants for RFC6962
LEAF_PREFIX = b'\x00'
NODE_PREFIX = b'\x01'

def sm3_hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

class MerkleTree:
    def __init__(self, leaves: List[bytes]):
        self.leaves = leaves
        self.tree: List[List[bytes]] = [[]]
        self.root: Optional[bytes] = None
        self._build_tree()

    def _build_tree(self):
        current_level = [sm3_hash(LEAF_PREFIX + leaf) for leaf in self.leaves]
        self.tree.append(current_level)

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                if i + 1 < len(current_level):
                    right = current_level[i+1]
                    node_hash = sm3_hash(NODE_PREFIX + left + right)
                else:
                    node_hash = sm3_hash(NODE_PREFIX + left + left)
                next_level.append(node_hash)
            current_level = next_level
            self.tree.append(current_level)
        
        if current_level:
            self.root = current_level[0]
        else:
            self.root = None
    
    def get_inclusion_proof(self, leaf_data: bytes) -> Optional[List[bytes]]:
        leaf_hash = sm3_hash(LEAF_PREFIX + leaf_data)
        try:
            leaf_index = self.leaves.index(leaf_data)
        except ValueError:
            return None # Leaf not found

        proof = []
        current_hash = leaf_hash
        current_index = leaf_index
        
        for level in self.tree[1:]:
            is_right_child = current_index % 2 == 1
            sibling_index = current_index - 1 if is_right_child else current_index + 1

            if sibling_index < len(level) and len(level) > 1:
                sibling_hash = level[sibling_index]
                proof.append(sibling_hash)
                
              
                current_index //= 2
                if is_right_child:
                    current_hash = sm3_hash(NODE_PREFIX + sibling_hash + current_hash)
                else:
                    current_hash = sm3_hash(NODE_PREFIX + current_hash + sibling_hash)
            elif len(level) == 1:
             
                if leaf_index == len(self.leaves) - 1 and len(self.leaves) % 2 == 1 and len(self.tree[-2]) % 2 == 1:
                    proof.append(current_hash) # The sibling is the hash of the leaf itself
                    current_hash = sm3_hash(NODE_PREFIX + current_hash + current_hash)
            
                pass
            
            if current_hash == self.root:
                break
        
        return proof

    def verify_inclusion_proof(self, leaf_data: bytes, proof: List[bytes]) -> bool:
        computed_hash = sm3_hash(LEAF_PREFIX + leaf_data)
        current_index = self.leaves.index(leaf_data)

        for sibling_hash in proof:
            is_right_child = current_index % 2 == 1
            if is_right_child:
                computed_hash = sm3_hash(NODE_PREFIX + sibling_hash + computed_hash)
            else:
                computed_hash = sm3_hash(NODE_PREFIX + computed_hash + sibling_hash)
            current_index //= 2

        return computed_hash == self.root


    def get_non_existence_proof(self, target_data: bytes) -> Optional[dict]:
        sorted_leaves = sorted(self.leaves)
        target_hash = sm3_hash(LEAF_PREFIX + target_data)

      
        prev_leaf, next_leaf = None, None
        for i, leaf in enumerate(sorted_leaves):
            leaf_hash = sm3_hash(LEAF_PREFIX + leaf)
            if leaf_hash > target_hash:
                next_leaf = leaf
                if i > 0:
                    prev_leaf = sorted_leaves[i-1]
                break
            if i == len(sorted_leaves) - 1:
                prev_leaf = sorted_leaves[i]

        proof = {}
        if prev_leaf:
            proof['prev_leaf'] = prev_leaf
            proof['prev_proof'] = self.get_inclusion_proof(prev_leaf)
        if next_leaf:
            proof['next_leaf'] = next_leaf
            proof['next_proof'] = self.get_inclusion_proof(next_leaf)
            
        return proof

    def verify_non_existence_proof(self, target_data: bytes, proof: dict) -> bool:
        target_hash = sm3_hash(LEAF_PREFIX + target_data)

     
        if 'prev_leaf' in proof and proof['prev_leaf'] == target_data:
            return False
        if 'next_leaf' in proof and proof['next_leaf'] == target_data:
            return False

       
        is_prev_valid = True
        if 'prev_leaf' in proof:
            is_prev_valid = self.verify_inclusion_proof(proof['prev_leaf'], proof['prev_proof'])
        
        is_next_valid = True
        if 'next_leaf' in proof:
            is_next_valid = self.verify_inclusion_proof(proof['next_leaf'], proof['next_proof'])

        if not (is_prev_valid and is_next_valid):
            return False

    
        if 'prev_leaf' in proof and 'next_leaf' in proof:
            prev_hash = sm3_hash(LEAF_PREFIX + proof['prev_leaf'])
            next_hash = sm3_hash(LEAF_PREFIX + proof['next_leaf'])
            return prev_hash < target_hash < next_hash
        elif 'prev_leaf' in proof:
            prev_hash = sm3_hash(LEAF_PREFIX + proof['prev_leaf'])
            return prev_hash < target_hash
        elif 'next_leaf' in proof:
            next_hash = sm3_hash(LEAF_PREFIX + proof['next_leaf'])
            return target_hash < next_hash
        
        return False


if __name__ == '__main__':
    # 10万个叶子节点
    leaves_data = [f'leaf_{i}'.encode('utf-8') for i in range(100000)]
    

    merkle_tree = MerkleTree(leaves_data)
    print(f"Merkle Tree Root: {merkle_tree.root.hex()}")
    

    target_leaf = leaves_data[50000]
    proof = merkle_tree.get_inclusion_proof(target_leaf)
    if proof:
        print(f"\nInclusion proof for 'leaf_50000' generated.")
        is_valid = merkle_tree.verify_inclusion_proof(target_leaf, proof)
        print(f"Inclusion proof is valid: {is_valid}")
    

    non_existent_leaf = b'non_existent_data'
    non_existence_proof = merkle_tree.get_non_existence_proof(non_existent_leaf)
    if non_existence_proof:
        print(f"\nNon-existence proof for 'non_existent_data' generated.")
        is_valid = merkle_tree.verify_non_existence_proof(non_existent_leaf, non_existence_proof)
        print(f"Non-existence proof is valid: {is_valid}")
