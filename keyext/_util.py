from itertools import combinations

def remove_duplicate(list):
    seen = set()
    seen_add = seen.add

    return [x for x in list if not (x in seen or seen_add(x))]


def ordered_combination(list, range):
    result = []
    for i in range:
        result.extend([
            [element for idx, element in sorted(subset)]
            for subset
            in combinations(enumerate(list), i)
        ])

    return result
