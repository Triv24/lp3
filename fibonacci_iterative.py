def fib(n):
  a = 0
  b = 1
  for i in range(0, n):
    c = a + b
    print(a)
    a = b
    b = c

num = int(input("Enter the number : "))
fib(num) 
      
