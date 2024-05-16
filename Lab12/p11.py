"""
        Item1   Item2   Item3   Item4   Item5
Alice     5       3       4       4       ?
User1     3       1       2       3       3
User2     4       3       4       3       5
User3     3       3       1       5       4
User4     1       5       5       2       1
"""

import numpy as np

# matricea de ratinguri
ratings = {
    'Alice': {'Item1': 5, 'Item2': 3, 'Item3': 4, 'Item4': 4},
    'User1': {'Item1': 3, 'Item2': 1, 'Item3': 2, 'Item4': 3, 'Item5': 3},
    'User2': {'Item1': 4, 'Item2': 3, 'Item3': 4, 'Item4': 3, 'Item5': 5},
    'User3': {'Item1': 3, 'Item2': 3, 'Item3': 1, 'Item4': 5, 'Item5': 4},
    'User4': {'Item1': 1, 'Item2': 5, 'Item3': 5, 'Item4': 2, 'Item5': 1},
}

# calc media ratinguri pe user
def average_ratings(ratings):
    avg_ratings = {}
    for user, items in ratings.items():
        avg_ratings[user] = np.mean(list(items.values()))
    return avg_ratings


# corelatia Pearson
def pearson_similarity(user1, user2, ratings):
    common_items = set(ratings[user1].keys()).intersection(set(ratings[user2].keys()))
    if len(common_items) == 0:
        return 0

    ratings1 = np.array([ratings[user1][item] for item in common_items])
    ratings2 = np.array([ratings[user2][item] for item in common_items])

    mean1 = np.mean(ratings1)
    mean2 = np.mean(ratings2)

    numerator = np.sum((ratings1 - mean1) * (ratings2 - mean2))
    denominator = np.sqrt(np.sum((ratings1 - mean1) ** 2)) * np.sqrt(np.sum((ratings2 - mean2) ** 2))

    if denominator == 0:
        return 0

    return numerator / denominator


# ratingul lipsa pentru Alice la Item5
def estimate_rating(user, item, ratings, avg_ratings):
    numerator = 0
    denominator = 0
    for other_user in ratings:
        if other_user != user and item in ratings[other_user]:
            sim = pearson_similarity(user, other_user, ratings)
            numerator += sim * (ratings[other_user][item] - avg_ratings[other_user])
            denominator += abs(sim)

    if denominator == 0:
        return avg_ratings[user]

    return avg_ratings[user] + numerator / denominator

def main():
    avg_ratings = average_ratings(ratings)
    print(f"Average ratings: {avg_ratings}")

    # ratingul estimat pentru Alice la Item5
    estimated_rating = estimate_rating('Alice', 'Item5', ratings, avg_ratings)
    print(f"Rating-ul estimat pe care Alice l-ar da produsului Item5 este: {estimated_rating:.4f}")


if __name__ == '__main__':
    main()