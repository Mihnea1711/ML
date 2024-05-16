"""
        Item1   Item2   Item3   Item4   Item5
Alice     5       3       4       4       ?
User1     3       1       2       3       3
User2     4       3       4       3       5
User3     3       3       1       5       4
User4     1       5       5       2       1
"""

import numpy as np
from pprint import pprint

# matricea de ratinguri
ratings = {
    'Alice': {'Item1': 5, 'Item2': 3, 'Item3': 4, 'Item4': 4},
    'User1': {'Item1': 3, 'Item2': 1, 'Item3': 2, 'Item4': 3, 'Item5': 3},
    'User2': {'Item1': 4, 'Item2': 3, 'Item3': 4, 'Item4': 3, 'Item5': 5},
    'User3': {'Item1': 3, 'Item2': 3, 'Item3': 1, 'Item4': 5, 'Item5': 4},
    'User4': {'Item1': 1, 'Item2': 5, 'Item3': 5, 'Item4': 2, 'Item5': 1},
}

# transf matr de ratinguri pentru a obtine o structura bazata pe produse
def transform_ratings(ratings):
    product_ratings = {}
    for user, items in ratings.items():
        for item, rating in items.items():
            if item not in product_ratings:
                product_ratings[item] = {}
            product_ratings[item][user] = rating
    return product_ratings

# calc sim cosinus intre 2 prod
def cosine_similarity(item1, item2, product_ratings):
    common_users = set(product_ratings[item1].keys()).intersection(set(product_ratings[item2].keys()))
    if len(common_users) == 0:
        return 0

    ratings1 = np.array([product_ratings[item1][user] for user in common_users])
    ratings2 = np.array([product_ratings[item2][user] for user in common_users])

    dot_product = np.dot(ratings1, ratings2)
    norm1 = np.linalg.norm(ratings1)
    norm2 = np.linalg.norm(ratings2)

    output = dot_product / (norm1 * norm2)

    if norm1 == 0 or norm2 == 0:
        return 0

    return output


# ratingul lipsa pentru Alice la Item5
def estimate_rating(user, item, ratings, product_ratings):
    numerator = 0
    denominator = 0
    for other_item in ratings[user]:
        # calc sim intre fiecare item care are rating si itemul dorit
        sim = cosine_similarity(item, other_item, product_ratings)
        numerator += sim * ratings[user][other_item]
        denominator += abs(sim)

    if denominator == 0:
        return 0

    return numerator / denominator

def main():
    product_ratings = transform_ratings(ratings)
    pprint(product_ratings)

    # calc ratingului estimat pentru Alice la Item5
    estimated_rating = estimate_rating('Alice', 'Item5', ratings, product_ratings)
    print(f"Rating-ul estimat pe care Alice l-ar da produsului Item5 este: {estimated_rating:.2f}")

if __name__ == '__main__':
    main()