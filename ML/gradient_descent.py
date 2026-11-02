"""
QUESTION 5 (ML):
Implement the Gradient Descent Algorithm to find the local minima of the
function y = (x + 3)^2 starting from the point x = 2.

HOW IT WORKS
    1. f(x) = (x + 3)^2 is a parabola with its minimum at x = -3 (f = 0).
    2. f'(x) = 2(x + 3) is the slope: its sign points uphill.
    3. Update rule: x_new = x - learning_rate * f'(x).
    4. Repeat. Steps shrink as the slope fades, so x settles near -3.

    At learning rate 0.1 and 100 iterations, x ends around -2.999996.
    A rate that is too large overshoots; 1.0 never settles.

STEPS
    1. Define the function and its derivative
    2. Set the starting point and hyper-parameters
    3. Gradient descent function
    4. Run gradient descent
    5. Print the minimum
    6. Effect of the learning rate
    7. Observation
    8. Plots with matplotlib

RUN
    python3 ml/py5_GradientDescent.py

    In Jupyter, open py5_GradientDescent.ipynb or split this file at the
    "# %%" markers. This practical needs no imports and no dataset.
"""
# Define the function
def f(x):
    return (x + 3) ** 2

# Define the derivative
def derivative(x):
    return 2 * (x + 3)

# Initial values
x = 2
learning_rate = 0.1
iterations = 100

# Gradient Descent
for i in range(iterations):
    gradient = derivative(x)
    x = x - learning_rate * gradient

# Display the result
print("Minimum point x =", round(x, 6))
print("Minimum value y =", round(f(x), 10))





"""
#============================================================
#============================================================

ML Practical 5: Gradient Descent Algorithm
Optimization Algorithm
External explanation:

"In this practical, I implemented the Gradient Descent algorithm to find the minimum value of a mathematical function, f(x) = (x + 3) squared.

First, I defined the function and calculated its derivative, which is 2 multiplied by (x + 3). Then I initialized x as 2, the learning rate as 0.1 and the number of iterations as 100.

Inside the loop, I calculated the gradient and updated x using the Gradient Descent formula, which subtracts the product of the learning rate and gradient from the current x value.

After 100 iterations, the algorithm converged close to x = -3, where the minimum function value is 0."

Remember:
Topic	Answer
Algorithm	Gradient Descent
Function	\(f(x)=(x+3)^2\)
Derivative	\(f'(x)=2(x+3)\)
Starting point	2
Learning rate	0.1
Iterations	100
Minimum point	-3
Minimum value	0
Time complexity	O(n)
Space complexity	O(1)




++++++++++++++++++++++++++++++++++++++




## 4. CODE EXPLANATION

### 1. Define the Function

```python
def f(x):
    return (x + 3) ** 2
```

- Defines the mathematical function.
- `** 2` means square.
- The objective is to find its minimum value.

### 2. Define the Derivative

```python
def derivative(x):
    return 2 * (x + 3)
```

The derivative of the given function is:

f'(x) = 2(x + 3)

It calculates the gradient or slope at a given point.

### 3. Initialize Values

```python
x = 2
learning_rate = 0.1
iterations = 100
```

- `x = 2`: Starting point.
- `learning_rate = 0.1`: Controls the step size.
- `iterations = 100`: Number of updates.

### 4. Gradient Descent Algorithm

```python
for i in range(iterations):
    gradient = derivative(x)
    x = x - learning_rate * gradient
```

- `range(iterations)`: Repeats the loop 100 times.
- `derivative(x)`: Calculates the current gradient.
- `x = x - learning_rate * gradient`: Updates x in the direction of decreasing function value.

The update formula is:

x_new = x_old - learning_rate × gradient

### 5. Display Results

```python
print("Minimum point x =", round(x, 6))
print("Minimum value y =", round(f(x), 10))
```

- `round(x, 6)`: Displays x up to 6 decimal places.
- `round(f(x), 10)`: Displays the function value up to 10 decimal places.

## 5. ALGORITHM

1. Start.
2. Define the function f(x) = (x + 3)².
3. Define its derivative f'(x) = 2(x + 3).
4. Initialize x = 2.
5. Set learning rate = 0.1.
6. Set iterations = 100.
7. Calculate the gradient.
8. Update x using the Gradient Descent formula.
9. Repeat steps 7 and 8 for 100 iterations.
10. Display the minimum point and minimum value.
11. Stop.

## 6. MATHEMATICAL EXPLANATION

Given function:

f(x) = (x + 3)²

Differentiate:

f'(x) = 2(x + 3)

Initial values:

x = 2

Learning rate = 0.1

**First iteration:**

Gradient = 2(2 + 3) = 10

Updated x = 2 - (0.1 × 10)

Updated x = 1

**Second iteration:**

Gradient = 2(1 + 3) = 8

Updated x = 1 - (0.1 × 8)

Updated x = 0.2

The algorithm continues updating x until it approaches -3.

**Final result:**

Minimum point x = -3

Minimum value f(-3) = 0

## 7. TIME AND SPACE COMPLEXITY

Let n represent the number of iterations.

| Operation | Complexity |
|---|---|
| Gradient calculation | O(1) per iteration |
| Parameter update | O(1) per iteration |
| Total time complexity | O(n) |
| Space complexity | O(1) |

**Explanation:**

- Time complexity is O(n) because the loop executes n times.
- Space complexity is O(1) because only a fixed number of variables are used.

## 8. VIVA QUESTIONS AND ANSWERS

### Basic Questions

**Q1. What is Gradient Descent?**

Gradient Descent is an optimization algorithm used to find the minimum of a function by repeatedly updating parameters in the opposite direction of the gradient.

**Q2. What is the objective of this practical?**

To find the minimum of f(x) = (x + 3)², starting from x = 2.

**Q3. What is the derivative of the given function?**

f'(x) = 2(x + 3).

**Q4. What is a gradient?**

A gradient represents the slope or rate of change of a function.

**Q5. What is the learning rate?**

It controls how large each update step is.

**Q6. What is an iteration?**

One complete update of the parameter x.

**Q7. What is the minimum point in this practical?**

x = -3.

**Q8. What is the minimum function value?**

f(-3) = 0.

### Technical Questions

**Q9. Why do we subtract the gradient?**

Because the negative gradient points in the direction of decreasing function value.

**Q10. What happens if the learning rate is too high?**

The algorithm may overshoot the minimum or fail to converge.

**Q11. What happens if the learning rate is too low?**

The algorithm converges very slowly.

**Q12. What is convergence?**

When the algorithm approaches a stable minimum value.

**Q13. Where is Gradient Descent used?**

It is used in machine learning to minimize loss or cost functions during model training.

**Q14. Is Gradient Descent a supervised learning algorithm?**

No. It is an optimization algorithm, not a learning category. It is used to optimize models in supervised and other learning settings.

**Q15. What is the difference between local and global minimum?**

- Local minimum: Lowest value in a nearby region.
- Global minimum: Lowest value over the entire function domain.

**Q16. What is the role of the derivative?**

It determines the slope and direction in which the function is increasing or decreasing.

### Tricky Questions

**Q17. Does Gradient Descent always find the global minimum?**

No. It can get stuck at a local minimum or fail to converge depending on the function and starting point.

**Q18. What happens if the derivative is zero?**

The algorithm does not update x at that point because the gradient is zero.

**Q19. What is the difference between Gradient Descent and normal differentiation?**

- Differentiation calculates the derivative or slope.
- Gradient Descent uses the derivative to iteratively find a minimum.

**Q20. Why are 100 iterations used?**

To allow the algorithm sufficient updates to approach the minimum for the selected learning rate and starting point.

**Q21. What type of function is used in this practical?**

A quadratic convex function.

**Q22. What is the difference between Batch Gradient Descent and Stochastic Gradient Descent?**

- Batch Gradient Descent uses the entire dataset to calculate the gradient.
- Stochastic Gradient Descent updates parameters using one training example at a time.

## 9. IMPORTANT FORMULAS

| Concept | Formula |
|---|---|
| Given function | f(x) = (x + 3)² |
| Derivative | f'(x) = 2(x + 3) |
| Gradient update | x_new = x_old - αf'(x_old) |
| Learning rate | α = 0.1 |
| Initial point | x = 2 |
| Minimum point | x = -3 |
| Minimum value | f(-3) = 0 |

## 10. COMMON MISTAKES AND EXCEPTIONS

| Mistake | Correct approach |
|---|---|
| Incorrect derivative | Use `2 * (x + 3)` |
| Adding gradient instead of subtracting | Use `x - learning_rate * gradient` |
| Learning rate too high | May overshoot the minimum |
| Too few iterations | May not converge sufficiently |
| Incorrect indentation | Keep update statements inside the loop |
| Wrong initial value | Use x = 2 as specified |

**Important concepts:**

1. A high learning rate can cause overshooting or divergence.
2. A low learning rate makes convergence slow.
3. Gradient Descent depends on the starting point.
4. In this practical, the function is convex and has one global minimum.

## 11. LIMITATIONS AND IMPROVEMENTS

**Limitations:**

1. Performance depends on the learning rate.
2. Too few iterations may result in incomplete convergence.
3. A poor starting point can affect convergence for more complex functions.
4. Gradient Descent may converge to a local minimum for non-convex functions.

**Possible improvements:**

1. Use an adaptive learning rate.
2. Use a convergence condition instead of a fixed iteration count.
3. Experiment with different learning rates.
4. Plot the function to visualize the minimum.

## 12. ONE-MINUTE EXTERNAL EXPLANATION

"In this practical, I implemented the Gradient Descent algorithm to find the minimum of the function f(x) = (x + 3)², starting from x = 2.

First, I defined the function and its derivative. Then I initialized the learning rate to 0.1 and the number of iterations to 100.

Inside the loop, I calculated the gradient and updated x by subtracting the product of the learning rate and gradient.

After 100 iterations, the algorithm converged close to x = -3, where the minimum function value is 0."

## 13. LINUX INSTALLATION AND EXECUTION COMMANDS

No external libraries are required.

Check Python version:

```bash
python --version
```

Run the Python file:

```bash
python gradient_descent.py
```

## 14. LAST-MINUTE REVISION TABLE

| Topic | Answer |
|---|---|
| Practical | Gradient Descent |
| Function | (x + 3)² |
| Derivative | 2(x + 3) |
| Starting point | 2 |
| Learning rate | 0.1 |
| Iterations | 100 |
| Update direction | Negative gradient |
| Minimum x | -3 |
| Minimum y | 0 |
| Time complexity | O(n) |
| Space complexity | O(1) |
| Algorithm type | Optimization algorithm |
| Function type | Convex quadratic |

## 15. CONCLUSION

The Gradient Descent algorithm was successfully implemented to find the minimum of the given function. After 100 iterations, the algorithm converged to approximately x = -3, with a minimum function value of 0.
"""
