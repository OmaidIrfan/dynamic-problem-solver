from flask import Flask, render_template, request, redirect, url_for, flash
app = Flask(__name__)
app.secret_key = "super secret key"

datasets = {
    'input01': 0,
    'input02': 0,
    'input03': 0,
    'input04': 0,
    'input05': 0,
    'input06': 0,
    'input07': 0,
    'input08': 0,
    'input09': 0,
    'input10': 0,
    'page': 0
}

result = {
    'algorithm': 0,
    'input': 0,
    'output': 0
}

##############################################################################################################################
# 
#                                               Algorithms
# 
############################################################################################################################### 

# Longest Common Subsequence
def algo1(X, Y, m, n, lookup):
    # return if we have reached the end of either string
    if m == 0 or n == 0:
        return 0
 
    # construct a unique dict key from dynamic elements of the input
    key = (m, n)
 
    # if sub-problem is seen for the first time, solve it and
    # store its result in a dict
    if key not in lookup:
 
        # if last character of X and Y matches
        if X[m - 1] == Y[n - 1]:
            lookup[key] = algo1(X, Y, m - 1, n - 1, lookup) + 1
 
        else:
            # else if last character of X and Y don't match
            lookup[key] = max(algo1(X, Y, m, n - 1, lookup),algo1(X, Y, m - 1, n, lookup))
 
    # return the sub-problem solution from the dictionary
    return lookup[key]

# Shortest Common Supersequence
def algo2(X, Y, m, n, lookup):
    # if we have reached the end of either sequence, return
    # length of other sequence
    if m == 0 or n == 0:
        return n + m
 
    # construct an unique dict key from dynamic elements of the input
    key = (m, n)
 
    # if sub-problem is seen for the first time, solve it and
    # store its result in a dict
    if key not in lookup:
 
        # if last character of X and Y matches
        if X[m - 1] == Y[n - 1]:
            lookup[key] = algo2(X, Y, m - 1, n - 1, lookup) + 1
 
        # else if last character of X and Y don't match
        else:
            lookup[key] = min(algo2(X, Y, m, n - 1, lookup),
                              algo2(X, Y, m - 1, n, lookup)) + 1
 
    # return the sub-problem solution from the dict
    return lookup[key]

# Levenshtein Distance (edit-distance)
def algo3(X, Y):
    (m, n) = (len(X), len(Y))
 
    # for all i and j, T[i,j] will hold the Levenshtein distance between
    # the first i characters of X and the first j characters of Y
    # note that T has (m+1)*(n+1) values
    T = [[0 for x in range(n + 1)] for y in range(m + 1)]
 
    # source prefixes can be transformed into empty by
    # dropping all characters
    for i in range(1, m + 1):
        T[i][0] = i                     # (case 1)
 
    # target prefixes can be reached from empty source prefix
    # by inserting every character
    for j in range(1, n + 1):
        T[0][j] = j                     # (case 1)
 
    # fill the lookup table in bottom-up manner
    for i in range(1, m + 1):
 
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:    # (case 2)
                cost = 0                # (case 2)
            else:
                cost = 1                # (case 3c)
 
            T[i][j] = min(T[i - 1][j] + 1,          # deletion (case 3b)
                          T[i][j - 1] + 1,          # insertion (case 3a)
                          T[i - 1][j - 1] + cost)   # replace (case 2 + 3c)
 
    return T[m][n]

# Longest Increasing Subsequence
def algo4(A):
    # list to store sub-problem solution. L[i] stores the length
    # of the longest increasing sub-sequence ends with A[i]
    L = [0] * len(A)
 
    # longest increasing sub-sequence ending with A[0] has length 1
    L[0] = 1
 
    # start from second element in the list
    for i in range(1, len(A)):
        # do for each element in sublist A[0..i-1]
        for j in range(i):
            # find longest increasing sub-sequence that ends with A[j]
            # where A[j] is less than the current element A[i]
            if A[j] < A[i] and L[j] > L[i]:
                L[i] = L[j]
 
        # include A[i] in LIS
        L[i] = L[i] + 1
 
    # return longest increasing sub-sequence (having maximum length)
    return max(L)

# Matrix Chain Multiplication (Order finding /paranthesization)
def algo5(dims, i, j, T):
    # base case: one matrix
    if j <= i + 1:
        return 0
 
    # stores minimum number of scalar multiplications (i.e., cost)
    # needed to compute the matrix M[i+1]...M[j] = M[i..j]
    min = float('inf')
 
    # if sub-problem is seen for the first time, solve it and
    # store its result in a lookup table
    if T[i][j] == 0:
 
        # take the minimum over each possible position at which the
        # sequence of matrices can be split
 
        """
            (M[i+1]) x (M[i+2]..................M[j])
            (M[i+1]M[i+2]) x (M[i+3.............M[j])
            ...
            ...
            (M[i+1]M[i+2]............M[j-1]) x (M[j])
        """
 
        for k in range(i + 1, j):
 
            # recur for M[i+1]..M[k] to get an i x k matrix
            cost = algo5(dims, i, k, T)
 
            # recur for M[k+1]..M[j] to get a k x j matrix
            cost += algo5(dims, k, j, T)
 
            # cost to multiply two (i x k) and (k x j) matrix
            cost += dims[i] * dims[k] * dims[j]
 
            if cost < min:
                min = cost
 
        T[i][j] = min
 
    # return min cost to multiply M[j+1]..M[j]
    return T[i][j]

# Partition-problem
def algo6(v, w, n, W, lookup):
    # base case: Negative capacity
    if W < 0:
        return float('-inf')
 
    # base case: no items left or capacity becomes 0
    if n < 0 or W == 0:
        return 0
 
    # construct an unique dict key from dynamic elements of the input
    key = (n, W)
 
    # if sub-problem is seen for the first time, solve it and
    # store its result in a dict
    if key not in lookup:
        # Case 1. include current item n in knapSack (v[n]) & recur for
        # remaining items (n - 1) with decreased capacity (W - w[n])
        include = v[n] + algo6(v, w, n - 1, W - w[n], lookup)
 
        # Case 2. exclude current item n from knapSack and recur for
        # remaining items (n - 1)
        exclude = algo6(v, w, n - 1, W, lookup)
 
        # assign max value we get by including or excluding current item
        lookup[key] = max(include, exclude)
 
    # return solution to current sub-problem
    return lookup[key]


def subsetSum(A, n, sum):
 
    # T[i][j] stores true if subset with sum j can be attained with
    # using items up to first i items
    T = [[False for x in range(sum + 1)] for y in range(n + 1)]
 
    # if sum is zero
    for i in range(n + 1):
        T[i][0] = True
 
    # do for ith item
    for i in range(1, n + 1):
 
        # consider all sum from 1 to sum
        for j in range(1, sum + 1):
 
            # don't include ith element if j-A[i-1] is negative
            if A[i - 1] > j:
                T[i][j] = T[i - 1][j]
            else:
                # find subset with sum j by excluding or including the ith item
                T[i][j] = T[i - 1][j] or T[i - 1][j - A[i - 1]]
 
    # return maximum value
    return T[n][sum]
 
 
# Return true if given list A[0..n-1] can be divided into two
# sublists with equal sum

# Partition-problem
def algo7(A):
 
    total = sum(A)
 
    # return true if sum is even and list can can be divided into
    # two sublists with equal sum
    return (total & 1) == 0 and subsetSum(A, len(A), total // 2)

# Rod Cutting Problem
def algo8(price, n):
        # T[i] stores maximum profit achieved from rod of length i
    T = [0] * (n + 1)
 
    # consider rod of length i
    for i in range(1, n + 1):
        # divide the rod of length i into two rods of length j
        # and i-j each and take maximum
        for j in range(1, i + 1):
            T[i] = max(T[i], price[j - 1] + T[i - j])
 
    # T[n] stores maximum profit achieved from rod of length n
    return T[n]

# Coin-change-making-problem
def algo9(S, N):
    # T[i] stores minimum number of coins needed to get total of i
    T = [0] * (N + 1)
 
    for i in range(1, N + 1):
 
        # initialize minimum number of coins needed to infinity
        T[i] = float('inf')
 
        # do for each coin
        for c in range(len(S)):
            # check if index doesn't become negative by including
            # current coin c
            if i - S[c] >= 0:
                res = T[i - S[c]]
 
                # if total can be reached by including current coin c,
                # update minimum number of coins needed T[i]
                if res != float('inf'):
                    T[i] = min(T[i], res + 1)
 
    # T[N] stores the minimum number of coins needed to get total of N
    return T[N]

# Word Break Problem
def algo10(dict, str, lookup):
    # n stores length of current substring
    n = len(str)
 
    # return true if we have reached the end of the String
    if n == 0:
        return True
 
    # if sub-problem is seen for the first time
    if lookup[n] == -1:
 
        # mark sub-problem as seen (0 initially assuming String
        # can't be segmented)
        lookup[n] = 0
 
        for i in range(1, n + 1):
            # consider all prefixes of current String
            prefix = str[:i]
 
            # if prefix is found in dictionary, then recur for suffix
            if prefix in dict and algo10(dict, str[i:], lookup):
                # return true if the can be segmented
                lookup[n] = 1
                return True
 
    # return solution to current sub-problem
    return lookup[n] == 1

@app.route('/')
def home():
    result['input'] = 0
    result['output'] = 0
    return render_template("APanel.html")

@app.route('/problem-solver/<page1>')
def problem_solver(page1):
    
    datasets['page'] = page1
    f = open(page1+".txt", "r")
    datasets['input01'] = f.readline()
    datasets['input02'] = f.readline()
    datasets['input03'] = f.readline()
    datasets['input04'] = f.readline()
    datasets['input05'] = f.readline()
    datasets['input06'] = f.readline()
    datasets['input07'] = f.readline()
    datasets['input08'] = f.readline()
    datasets['input09'] = f.readline()
    datasets['input10'] = f.readline()
    f.close()

    if page1 == 'p01':
        result['algorithm'] = 'Longest Common Subsequence'
    elif page1 == 'p02':
        result['algorithm'] = 'Shortest Common Supersequence'
    elif page1 == 'p03':
        result['algorithm'] = 'Levenshtein Distance (edit-distance)'
    elif page1 == 'p04':
        result['algorithm'] = 'Longest Increasing Subsequence'
    elif page1 == 'p05':
        result['algorithm'] = 'Matrix Chain Multiplication (Order finding /paranthesization)'
    elif page1 == 'p06':
        result['algorithm'] = '0-1-knapsack-problem'
    elif page1 == 'p07':
        result['algorithm'] = 'Partition-problem'
    elif page1 == 'p08':
        result['algorithm'] = 'Rod Cutting Problem'
    elif page1 == 'p09':
        result['algorithm'] = 'Coin-change-making-problem'
    else:
        result['algorithm'] = 'Word Break Problem'

    return render_template("problem-solver.html", datasets = datasets, result = result, algo = result['algorithm'])

@app.route('/getAnswer',  methods = ['POST'])
def getAnswer():
    i01 = request.form['input-01']
    i02 = request.form['input-02']
    i03 = request.form['input-03']
    i04 = request.form['input-04']
    i05 = request.form['input-05']
    i06 = request.form['input-06']
    i07 = request.form['input-07']
    i08 = request.form['input-08']
    i09 = request.form['input-09']
    i10 = request.form['input-10']

    page = datasets['page']

    if i01 == '1':
        result['input'] = datasets['input01']
    elif i02 == '1':
        result['input'] = datasets['input02']
    elif i03 == '1':
        result['input'] = datasets['input03']
    elif i04 == '1':
        result['input'] = datasets['input04']
    elif i05 == '1':
        result['input'] = datasets['input05']
    elif i06 == '1':
        result['input'] = datasets['input06']
    elif i07 == '1':
        result['input'] = datasets['input07']
    elif i08 == '1':
        result['input'] = datasets['input08']
    elif i09 == '1':
        result['input'] = datasets['input09']
    elif i10 == '1':
        result['input'] = datasets['input10']
    else:
        flash('Looks like you have not selected an input!')
        return render_template("problem-solver.html", datasets = datasets, result = result, algo = result['algorithm'])

    if page == 'p01':
        modified_data = result['input'].split(",")
        modified_data1 = modified_data[1].replace('\n', '')
        lookup = {}
        result['output'] = f'The length of LCS is {algo1(modified_data[0], modified_data1, len(modified_data[0]), len(modified_data1), lookup)}.'
    elif page == 'p02':
        modified_data = result['input'].split(",")
        modified_data1 = modified_data[1].replace('\n', '')
        lookup = {}
        result['output'] = f'The length of shortest Common supersequence is {algo2(modified_data[0], modified_data1, len(modified_data[0]), len(modified_data1), lookup)}.'
    elif page == 'p03':
        modified_data = result['input'].split(",")
        modified_data1 = modified_data[1].replace('\n', '')
        result['output'] = f'The Levenshtein Distance is {algo3(modified_data[0], modified_data1)}.'
    elif page == 'p04':
        modified_data = result['input'].split(",")
        for item in range(0,len(modified_data)-1):
            modified_data[item] = int(modified_data[item])
        modified_data.pop()
        result['output'] = f'Length of LIS is {algo4(modified_data)}.'
    elif page == 'p05':
        modified_data = result['input'].split(",")
        for item in range(0,len(modified_data)-1):
            modified_data[item] = int(modified_data[item])
        modified_data.pop()
        T = [[0 for x in range(len(modified_data))] for y in range(len(modified_data))]
        result['output'] = f'Minimum cost is {algo5(modified_data, 0, len(modified_data) - 1, T)}.'
    elif page == 'p06':
        modified_data = result['input'].split(",")
        for item in range(0,len(modified_data)-1):
            modified_data[item] = int(modified_data[item])
        modified_data.pop()
        if(len(modified_data) % 2 != 0):
            modified_data.append(34)
        w = [0] * (len(modified_data) // 2)
        v = [0] * (len(modified_data) // 2)
        temp = 0
        for item in range(0,len(modified_data)-1,2):
            v[temp] = modified_data[item]
            temp = temp + 1
        temp = 0
        for item in range(1,len(modified_data)-1,2):
            w[temp] = modified_data[item]
            temp = temp + 1
        W = 143
        lookup = {}
        result['output'] = f'Knapsack value is {algo6(v, w, len(v) - 1, W, lookup)}.'
    elif page == 'p07':
        modified_data = result['input'].split(",")
        for item in range(0,len(modified_data)-1):
            modified_data[item] = int(modified_data[item])
        modified_data.pop()
        result['output'] = f'Is partition possible? {algo7(modified_data)}.'
    elif page == 'p08':
        modified_data = result['input'].split(",")
        for item in range(0,len(modified_data)-1):
            modified_data[item] = int(modified_data[item])
        modified_data.pop()
        n = 76
        result['output'] = f'Profit is {algo8(modified_data, n)}.'
    elif page == 'p09':
        modified_data = result['input'].split(",")
        for item in range(0,len(modified_data)-1):
            modified_data[item] = int(modified_data[item])
        modified_data.pop()
        N = 76
        result['output'] = f'Minimum number of coins required to get desired change is {algo9(modified_data, N)}.'
    elif page == 'p10':
        modified_data = result['input'].split(",")
        str1 = "muhammadumar"
        lookup = [-1] * (len(str1) + 1)
        if algo10(modified_data, str1, lookup):
            result['output'] = "String can be segmented"
        else:
            result['output'] = "String can't be segmented"
    else:
        flash('Please try again!')
    return render_template("problem-solver.html", datasets = datasets, result = result, algo = result['algorithm'])