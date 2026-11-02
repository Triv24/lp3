class Item:
    def __init__(self, weight, profit):
        self.weight = weight
        self.profit = profit
        self.ratio = profit / weight

def bound(level, profit, weight, capacity, items):
    if weight >= capacity:
        return 0
    max_profit = profit
    total_weight = weight
    i = level + 1
    while i < len(items) and total_weight + items[i].weight <= capacity:
        total_weight += items[i].weight
        max_profit += items[i].profit
        i += 1
    if i < len(items):
        max_profit += (capacity - total_weight) * items[i].ratio
    return max_profit

def knapsack(weights, profits, capacity):
    n = len(weights)
    items = [Item(weights[i], profits[i]) for i in range(n)]
    items.sort(key=lambda x: x.ratio, reverse=True)
    max_profit = 0
    queue = [(-1, 0, 0, bound(-1, 0, 0, capacity, items))]
    while queue:
        level, profit, weight, upper_bound = queue.pop(0)
        if upper_bound <= max_profit:
            continue
        next_level = level + 1
        if next_level >= n:
            continue
        item = items[next_level]
        new_weight = weight + item.weight
        new_profit = profit + item.profit
        if new_weight <= capacity:
            max_profit = max(max_profit, new_profit)
            new_bound = bound(
                next_level, new_profit, new_weight, capacity, items
            )
            if new_bound > max_profit:
                queue.append((
                    next_level, new_profit, new_weight, new_bound
                ))
        new_bound = bound(
            next_level, profit, weight, capacity, items
        )
        if new_bound > max_profit:
            queue.append((
                next_level, profit, weight, new_bound
            ))
    return max_profit

n = int(input("Enter number of items: "))
weights = []
profits = []
for i in range(n):
    w = int(input("Enter weight: "))
    p = int(input("Enter profit: "))
    weights.append(w)
    profits.append(p)
capacity = int(input("Enter knapsack capacity: "))
result = knapsack(weights, profits, capacity)
print("Maximum Profit =", result)
