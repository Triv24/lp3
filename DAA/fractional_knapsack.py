def fractional_knapsack(weights, profits, capacity):
    n = len(weights)
    ratio = []
    for i in range(n):
        ratio.append(profits[i] / weights[i])
    for i in range(n):
        for j in range(i + 1, n):
            if ratio[i] < ratio[j]:
                ratio[i], ratio[j] = ratio[j], ratio[i]
                weights[i], weights[j] = weights[j], weights[i]
                profits[i], profits[j] = profits[j], profits[i]
    total_profit = 0
    for i in range(n):
        if capacity >= weights[i]:
            total_profit += profits[i]
            capacity -= weights[i]
        else:
            total_profit += ratio[i] * capacity
            break
    return total_profit

n = int(input("Enter number of items: "))
weights = []
profits = []

for i in range(n):
    w = float(input("Enter weight of item: "))
    p = float(input("Enter profit of item: "))
    weights.append(w)
    profits.append(p)

capacity = float(input("Enter knapsack capacity: "))
max_profit = fractional_knapsack(weights, profits, capacity)
print("Maximum Profit =", max_profit)
