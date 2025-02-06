from typing import List
from tqdm import tqdm
from functools import total_ordering
from sortedcontainers import SortedDict, SortedList, SortedSet


@total_ordering
class Item:
    def __init__(self, pair: tuple[tuple, int]):
        self.pair = pair

    def __hash__(self):
        return hash(self.pair[0])

    def __eq__(self, other):
        return self.pair[1] == other.pair[1]

    def __lt__(self, other):
        return self.pair[1] < other.pair[1]

    def __repr__(self):
        return f"{self.pair}"


class N_Gram:
    def __init__(self, n):
        self.n = n  # medusa_head + 1(backbone)

    def preload(self, token_list: List[int]):
        self.lookup = dict()
        for i in tqdm(range(len(token_list) - self.n + 1), desc="preloading"):
            self.update(token_list[i : i + self.n])

    def clear(self):
        self.lookup = dict()

    def update(self, n_gram: List[int]):
        assert len(n_gram) == self.n
        cur_token = n_gram[0]
        continuation = tuple(n_gram[1 : self.n])

        if cur_token not in self.lookup:
            self.lookup[cur_token] = {
                "appearance_count": dict({continuation: 1}),
                "sorted_set": SortedSet([Item((continuation, 1))]),
            }
        else:
            cur_value = self.lookup[cur_token]["appearance_count"].get(continuation, 0)
            self.lookup[cur_token]["appearance_count"][continuation] = cur_value + 1
            if cur_value > 0:
                self.lookup[cur_token]["sorted_set"].remove(Item((continuation, cur_value)))
            self.lookup[cur_token]["sorted_set"].add(Item((continuation, cur_value + 1)))

    def topk(self, token: int, k: int):
        result = []
        if token not in self.lookup:
            return result
        for i, continuation in enumerate(reversed(self.lookup[token]["sorted_set"])):
            if i >= k:
                break
            result.append((token,) + continuation.pair[0])
        return result


if __name__ == "__main__":
    from medusa_choices import full_tree

    flattened = [item for sublist in full_tree for item in sublist]
    n_gram = N_Gram(4)
    n_gram.preload(flattened)
    print(n_gram.topk(0, 5))
    print(n_gram.lookup[0]["sorted_set"])

    # print(len(set( [Item(((1,2,3,), 3)),  Item(((1,2,3,), 3))] )))
