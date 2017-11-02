
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *p1(l1), *p2(l2);
		ListNode *res = new ListNode(-1);
		int carry = 0;
		while(p1 || p2){
			int num1 = p1? p1->val : 0;
			int num2 = p2? p2->val : 0;
			int sum = num1 + num2 + carry
			res -> next = new ListNode(sum % 10);
			carry = sum / 10;
			if (p1) p1 = p1 -> next;
			if (p2) p2 = p2 -> next;
		}
		if (carry) res -> next = new ListNode(1);
		return res;
    }
};

 
class exception{
public:
	exception() noexcept;
	exception(const exception& e) noexcept;
	virtual ~exception() noexcept;
	virtual exception& operator = (const exception& e) noexcept;
	virtual const char* what() noexcept;
}

class runtime_error: public exception{
public:
	explicit runtime_error(const string& msg) noexcept;
	explicit runtime_error(const char* msg) noexcept;
}

since the functionality of exceptions is to give some useful information for debugging, 
why do we need to define our own exception classes?



#inlcude <thread>
#include <vector>
using namespace std;

class Counter{
private:
	int value;
	std::mutex m;
public:
	Counter():value(0){}
	void increment(){
		m.lock();
		value++;
		m.unlock();
	}
};

int main(){
	Counter counter;
	vector<thread> threads;
	for(int i = 0; i < 5; i++){
		threads.push_back(thread([&counter](){
			for(int j = 0; j < 100; j++)
				counter.increment();
		}))
	}
	
	for(auto& e: threads) e.join();
	
}


read the value of value 
increment 1
write to value

problem: before you write the result to the value, another thread might be writing to the value, so we lost some 
not atomic
multiple threads might read the same value, losing some increment
interleaving

#include <mutex>
std::mutex m;
m.lock();

m.unlock();
only one object can have the lock

void decrement(){
	lock_guard<mutex> guard(m);
	m.lock();
	value--;
	m.unlock();
}

std::lock_guard
mutex can only be acquired once by the same thread


std::recursive_mutex mutex;

void mul(int x){
	std::lock_guard<std::mutex> guard(mutex);
	i *= x;
}

void div(int x){
	std::lock_guard<std::recursive_mutex> guard(mutex);
	i /= x;
}

void both(int x){
	std::lock_guard<std::recursive_mutex> guard(mutex);
	mul(x);
	div(x);
}

otherwise deadlock
prevent deadlock: make sure threads acquire the locks in an agreed order
make sure mutexes are unlocked properly

race condition: accessing shared data without an agreed order

priority inversion: a thread with higher priority should wait for low one that holds a lock for which the higher one is waiting

condition number



	
class Solution {
public:
    bool canFinish(int num, vector<pair<int, int>>& prerequisites) {
        vector<bool> visited(num, false);
        for(int i = 0; i < num; i++)
            if (!visited[i] && !canFinish(num, prerequisites, i, visited))
                return false;
        return true;
    }
};

import pandas as pd
pd.Series?

initialization of a pandas seires:

pd.Series(['tiger','dog','cat'])

pd.Series([1,2,None])

dtype
object 
int64
float64

import numpy as np
np.nan

s = pd.Series([1,2,3,4,5], index = ['a', 'b', 'c', 'd', 'e'])


sports = {10: 'a', 11: 'b', 12: 'c', 13: 'd'}
s = pd.Series(sports)
s[key] can work, but when the indices are not ordered from 0 then it would create a mess

s.iloc[]
s.loc[]

np.sum is faster than sum

np.sum(s) adds up all the values in the Series

vectorization is much faster than iterative
%%timeit -n 100
for a cel
s+=2


for label, value in s.iteritems():
	s.set_value(label, value+2)
equivalent to s+=2

s = pd.Series(np.random.randint(0,10000,10000))

concatenate two series

s1 
s2
s1.append(s2)

it does not change neither s1 not s2, instead it creates another series

each series is a row of a table

purchase_1 = pd.Series({'Name': 'Chris', 
						'Item purchased': 'Dog food',
						'Cost': 2.5})
					
df = pd.Dataframe([purchase_1, purchase_2, purchase_3], index = ['Store1', 'Store2', 'Store2'])
df.head()


names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index) 
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')
df.head()



Two sigma:
--- graph search and dynamic programming
--- construct a median heaps  
--- friend circle question, can be found on leetcode
--- the difference between a thread and a process.  
Solve Gaussian quantile by newtons method
Remove minimum number of paranthesis to make expression valid ([{
Uniformly sample the superset of a set of integers (solved by uniform sampling then sort), the input is a vector and the output is vector of vectors
Find the number of integer solutions to a linear inequality equation ( recursion)
Given two sorted vectors, find a threshold, if greater categorize to A otherwise to B. Find minimum classification errors (use binary search. Actually use a binary search on top of a binary search is even better)
Merge sort, but each merge are k to 1 instead of 2 to 1, heap)
Three people play a game, 2 color hats, can only see other people's black or white hat. If at least one wins and nobody loses, win. what's your strategy. (75%, for more general case, look at hamming code)
If anual sharpe 8 but loses money, how many days needed to tell statistical significance.
How do you tell if a stock delta price series is reverted in time ( volatility increase sharply and decrease slowly)
How to do weighted linear regression (just weight the terms in summation...)  

--- What is a template? Can you separate the interface and the implementation parts, why?  
	What is a virtual function and how it works?  vtable vpointer

--- power of 4
	boolean powerOf4(long v) {
        return ((v & (v - 1)) == 0) && ((v & 0x55555555) != 0);
    }
--- Two dynamic programming questions.  
--- priority queue implementation. 
	where to include build-heap, insert, deletion, top, pop
--- Singleton and factory pattern
--- hashtable implementation
--- Implement a merge sort in place
--- Imagine a game with a sequence of numbers. Each player can take either the head or tail element on any given turn. The player adds that number to their total score. Assuming both players play optimally, what's the maximum score that can be achieved by any player?  
--- Define an objective function for a factor-based quantitative portfolio model.  
--- find  the longest string chain possible given a dictionary of words.
--- regular expression matching
	write a function to auto generate test cases
--- postfix notation calculator
--- string compression
--- Pros/cons of merge sort and quick sort
--- find all palindromes in a string
--- extend your implementation of reverse Polish notation to include more operators
--- Find max benefit from 2 transactions given some stock prices
--- LRU
--- Explain the software development cycle.  
--- Given a portfolio made up of equal parts of two assets that have covariance of y and each of which have x volatility, what is the overall portfolio volatility?  
--- Make sure you can solve those at difficult level on HackRank
--- write a string class in C++
--- How many publicly traded stocks do you think there are?  
--- Bitwise Operation, Add one to a number without using +
--- Find all the subsets of size k 
--- You have two time series Xt and Yt. You do a regression of Y over X and get coefficient \beta, t-stats and R2. Now you double the size of the original series with original data and do the regression with new \beta, t-stats and R2. How will \beta, t-stats and R2 change?
--- given an array of size n, find the fastest way to find maximum and minimum element. 
--- Two questions for the coding test, one involving pure coding and the other involving some data analysis.




singleton class: only one instanciation can be created.

class singleton{
private:
		singleton(){} // disallow default constructor
		singleton(const singleton& other){} // disallow copy constructor
		singleton& operator=(const singleton& other){} // disallow assignment operator
		static singleton* obj;
		static std::mutex mutex; 
public:
	static singleton* getinstance(){
		mutex.lock(); // this is slow because of the overhead of creating a lock
		if (obj == nullptr){
			obj = new singleton();
			return obj;
		}
		else 
			return obj;
		mutex.unlock();
	}
};

singleton* single = singleton::getinstance();
singleton* single2 = singleton::getinstacne();

is it thread safe? what happens when two or more thread are trying to create an instance of this singleton class?



class Solution {
public:
    int calPoints(vector<string>& ops) {
        stack<int> points;
		int res = 0;
		for(string s: ops){
			if (s == "C"){// last round invalid
				int last = points.top();
				res -= last;
				points.pop();
			}
			else if (s == "D")
		}
    }
};



union and find
union represents adding one edge between two nodes or two groups
find means query if there is an edge between two parties.

synchronization primitives
atomicity is unbreakability
uninterrupted opration



mutex is locking mechanism used to synchronize access to a resource
only one task (thread or process) can acquire the mutex and has the ownership

recursive mutex has a count associated with it and it should be unlocked as many times as it has been locked.
a non-recursive mutex is locked more than once by the same thread or process it will cause a deadlock

binary semaphore and mutex are not the same, one is sigaling mechanism and the other is locking mechanism.
a programmer prefers mutex to semaphore with count 1

while loading data from a disk, the cpu does not wait on it and instead executing some other tasks,
when the data is ready from the disk, the interrupt service routine is waken up and return the data to the requester. 

it has to be very quick as there might be multiple interrupt requested and if one is masked it might cause data loss. 
program counter is contains the address of the instruction being executed at the current time, it is increased by 1 when
each instruction gets fetched. 

register is a quickly accessible location to cpu 
threads within the same process share the same code section, data section, files and signals.

there are two types of threads; use thread and kernel thread
arrival time, completion time, turn around time (diff between completion and arrival), burst time (time required by a process for CPU), wait time is diff between burst and turn around time

highest response ratio next HRRN prevents starvation, wait time / burst time

first come first serve can cause long waiting times, espicailly when the first few jobs take too much cpu time.
shortest job first and shortest remaining time first can cause starvation, a long process waiting while short processes keep coming

critical section is the code block surrounded by a lock

deadlock can rise when among a set of processes, at least one process is holding a resource and at the same time waiting for another resource which is held by another process 
that is waiting for that thread. and it is circular wait.

Thread A holds resource A and it is waiting for resource B which is held by thread B, at the same time thread B is waiting for resource A. 
if a deadlock is very rare then let it happen and then reboot the system like what happens in both windows and unix system.



class Solution {
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int N = edges.size();
		vector<int> ids(N+1, 0);
		vector<int> sz(N+1, 1);
		vector<int> res(2, 0);
		for(int i = 0; i < N; i++){
			int a = edges[i][0], b = edges[i][1];
			if (!find(ids,a,b)) unite(ids,sz,a,b);
			else res = edges[i];
		}
		return res;
    }
private:
	bool find(const vector<int>& ids, int a, int b){
		return root(ids,a) == root(ids, b);
	}
	
	int root(const vector<int>& ids, int i){
		while(i != ids[i]){
			ids[i] = ids[ids[i]]; // path compression by pointing to grandparent
			i = ids[i];
		}
		return i;
	}
	
	void unite(vector<int>& ids, vector<int>& sz, int a, int b){
		if (sz[a] < sz[b])
			ids[a] = root(ids, b);
		else 
			ids[b] = root(ids, a);
	}
};

the address a program uses is not physical address.
The addresses generated by a program are automatically translated to physical address.
It is mapped dynamically at run time and it may occupy different parts of physical address. 

001100011110101110101110101

class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int pos = 1;
		for(int i = 1; i < nums.size(); i++){
			if (nums[i] != nums[i-1])
				nums[pos++] = nums[i];
		}
		nums.erase(nums.begin() + pos + 1, nums.end() );
		return nums;
    }
};



class Solution {
public:
	vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
		vector<vector<int>> res;
		if (candidates.size() == 0) return res;
		sort(candidates.begin(), candidates.end());
		return cSum(candidates, target, 0);
	}

	vector<vector<int>> cSum(const vector<int>& nums, int target, int startInd) {
		vector<vector<int>> res;
		if (nums[startInd] > target) return res;
		for (int i = startInd; i < nums.size(); i++) {
			if (nums[i] == target) {
				vector<int> oneSol(1, nums[i]);
				res.push_back(oneSol);
				continue;
			}
			if (nums[i] < target) {
				auto subRes = cSum(nums, target - nums[i], i);
				for (int j = 0; j < subRes.size(); j++) {
					subRes[j].insert(subRes[j].begin(), nums[i]);
					res.push_back(subRes[j]);
				}
			}
			else break;
		}
		return res;
	}
};


class Solution {
public:
    bool canJump(vector<int>& nums) {
        int start = 0, end = 0;
		int maxReach = 0;
		while(start <= maxReach){
			for(int i = start; i <= end; i++){
				if (i + nums[i] > maxReach) maxReach = i + nums[i];
				if (maxReach > n - 1) return true;
			}
			start = end + 1;
			end = maxReach;
		}
		return false;
    }
};

class Solution {
public:
    int uniquePaths(int m, int n) {
		return countPaths(m,n,0,0);
    }
	int countPaths(int m, int n, int m0, int n0){
		if (m0 == m && n0 == n) return 1;
		return countPaths(m, n, m0 + 1,n0) + countPaths(m,n,m0,n0+1);
	}
};

int a = 3;
void *p=&a;
int *intp = static_cast<int*>(p);


unary operator @ applied to an object x
as a member of class x: x.operator@()
or as a non-member function
operator@(x)

binary operator @ applied to x and y
as a member of the left operand's member 
x.operator@(y)
or as a non-member operator@(x,y)

non-member operators are usually friend of their operands type

always provide a set of related operators
if you overload + the user would expect to use += 
++a and a++
a>b and a<b
copy and assignment

array subscription and assignemnt must be member
<< and >> must be friend functions instead of member
because the left operand of them is ostream which cannot be modified
Rules:
unary operator always member
binary: if both operands are treated equally, non-member
else member of the left


assignment operator
X& X::operator=(X other)

std::ostream& operator<<(std::ostream& os, const X& x){
	//write to the stream
	return os;
}

std::istream& operator>>(std::istream& is, X& x){
	// write to x
	return is;
}

== != < > <= >= should be none members

inline bool operator==(const X& l, const X& r){}
inline bool operator!=(const X& l, const X& r){return !operator==(l,r);}
inline bool operator<(const)

post increment implementation

X& operator++(int){
	X temp(*this);
	operator++();
	return temp;
}

rule of three:
if you need any of the three you probably need all of them
copy constructor
assignment operator
destructor
for managing resource

this != &other // not good in practice

while( a != b){
    	//for the end of first iteration, we just reset the pointer to the head of another linkedlist
        a = a == null? headB : a.next;
        b = b == null? headA : b.next;    
    }
    
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
		int n = nums.size();
		if (!n) return 0;
        vector<int> res(n, 1);
		for(int i = 1; i < n; i++){
			for(int j = 0; j < i; j++){
				if (nums[i] > nums[j])
					res[i] = max(res[i], res[j] + 1);
			}
		}
		return res[n-1];
    }
};


typedef vector<vector<int>> vvint;

class Solution {
public:
    int longestIncreasingPath(const vvint& matrix) {
        int m(matrix.size());
        if (!m) return 0;
        int n(matrix[0].size());
        if (!n) return 0;
        vvint dp(m,vector<int>(n,0));
        
        int maxl = 0;
        for(int i=0; i<m;++i)
            for(int j=0; j<n; ++j){
                dp[i][j] = longest(i,j,matrix,dp);
                if (maxl < dp[i][j]) maxl = dp[i][j];
            }
        
       return maxl;
    }
    
    int longest(int i, int j,const vvint& matrix,vvint& dp){
        if (dp[i][j] > 0) return dp[i][j];
        int max = 0;
        int m(matrix.size());
        int n(matrix[0].size());
        if (i>=1 && matrix[i-1][j] > matrix[i][j]){
            dp[i-1][j] = longest(i-1,j,matrix,dp);
            int temp = 1 + dp[i-1][j];
            if (temp > max) max = temp;
        }
        
        if (i+1<m && matrix[i+1][j] > matrix[i][j]){
            dp[i+1][j] = longest(i+1,j,matrix,dp);
            int temp = 1 + dp[i+1][j];
            if (temp > max) max = temp;
        }
        
        if (j>=1 && matrix[i][j-1] > matrix[i][j]){
            dp[i][j-1] = longest(i,j-1,matrix,dp);
            int temp = 1 + dp[i][j-1];
            if (temp > max) max = temp;
        }
        
        if (j+1<n && matrix[i][j+1] > matrix[i][j]){
            dp[i][j+1] = longest(i,j+1,matrix,dp);
            int temp = 1 + dp[i][j+1];
            if (temp > max) max = temp;
        }
        
        return (max>1)?max:1;
    }
    
};




vector<vector<int>> dir = {{-1, 0},{1, 0},{0, -1},{0, 1}}; // up, down, left, right



 
class Solution {
public:
    bool increasingTriplet(vector<int>& a) {
        int n = a.size();
		if (n <= 2) return false;
		int min1 = a[0], min2=INT_MAX;
		// min1 is the current min
		// min2 is the current min that is greater than at least one number
		for(int i = 1; i < n; i++){
		    if (a[i] < min1) { min1 = a[i]; continue;}
		    if (a[i] > min2) return true;
		    if (a[i] > min1 && a[i] < min2) min2 = a[i];
		}
		return false;
    }
};


summary of huffman encoding

class Solution {
public:
	typedef array<int, 3> Int3;
	struct comp{ // functor(function object)
		bool operator()(Int3 p1, Int3 p2){
			return (p1[0] + p1[1]) > (p2[0] + p2[1]);
		}
	};
    vector<pair<int, int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
		int n1 = nums1.size(), n2 = nums2.size();
		vector<int> ind(n1, 0);
		// build the heap with candidate pairs
		priority_queue<Int3, vector<Int3>, comp> minHeap;
		for(int i = 0; i < n1; i++){
			
		}
		
		vector<IntPair> res(k);
		int count = 0;
		while(count < k){
			
		}
    }
};

class Solution {
public: // ACGT only
	typedef pair<string, unsigned int> node;
	int minMutation(string start, string end, vector<string>& bank) {
		string acgt = "ACGT";
		// set up the hash map for gene bank
		unordered_map<string, bool> geneMap;
		for (int i = 0; i<bank.size(); i++) geneMap[bank[i]] = true;

		queue<node> trace;
		trace.push(make_pair(start, 0));
		unordered_map<string, bool> record; // record all the intermediate strings that appeared in the trace
		record[start] = true;
		while (!trace.empty()) {
			auto cur = trace.front();
			if (cur.first == end) return cur.second;
			trace.pop();
			// now expand each char in cur
			for (int i = 0; i < 8; i++) {// loop through every char in the gene
				string temp(cur.first);
				for (int j = 0; j < 4; j++) {// loop through every possible mutation on each char 
					temp[i] = acgt[j];
					if (geneMap[temp] && !record[temp]) {// if the new gene is in the bank and not appeared in the trace before
						trace.push(make_pair(temp, cur.second + 1));
						record[temp] = true;
					}
				}
			}
		}
		return -1;
	}
};

design patterns show relationships between classes and objects
elements of reusable object oriented software


builder pattern:
when we want to create complex objects and do not want a constructor with so many arguments we use an intermediate class as the builder
for example:
class Pizza{
	private:
		double size;
		double topping......
		some other members
	public:
		void setSize()......
};

to build a pizza object, Pizza(............)

or define a builder which is virtual 

class PizzaBuilder{
	public:
		virtual void set() = 0; // pure virtual function without implementation
	protected:
		unique_pointer<Pizza> m_pizza; // why protected? inherited by concrete classes
};

class HawaiianPizzaBuilder: public PizzaBuilder{
	void set(){m_pizza->}
};

delete an object with its base class pointer; virtual destructor

enum types can have overloaded operators 
enum color{red, green, black};
std::ostream& operator<<(std::ostream& os, const color& c){
	
}

function pointer:
int f(int a){
	
}

int (*f)(int); // this is defining f as a pointer to a function that takes in an int and returns an int

typedef typedef_declaration; 

in declarations, the name is an instance of the type
int i;
int (*f)(int,int);
int a[3];

but when preceded by "typedef" the name is an alias of the type
typedef int i;
typedef int (*f)(int,int); // f now represents the type pointer to a function that takes in two int and returns an int
typedef int a[3]; // a now represents an array of 3 int

#define NULL (void*)0 // 0 cast to a generic pointer which can be converted to any type
nullptr to save: it can be implicitly converted into any pointer type but not integral types except for bool
it is a pointer literal of type std::nullptr_t, you cannot take the address of nullptr

for graph(V,E) represented by adjacency list
breadth first traversal takes O(V + E) in time
depth first traversal takes O(V + E) in time

when a graph is dense, adjacency list loses its advantage, takes space O(V^2)
matrix: O(V^2), list: O(V + E)

course schedule; detect cycle

class Solution {
public:
    bool canFinish(int num, vector<pair<int, int>>& prerequisites) {
        vector<vector<int>> graph(num, vector<int>{});
		// build the adjacency list using edges
		for(int i = 0; i < prerequisites.size(); i++){
			graph[prerequisites[i].first].push_back(prerequisites[i].second);
		}
		// check cycle with DFS
		vector<bool> visited(num, false), curPath(num, false);
		for(int i = 0; i < num; i++){
			if (!visited[i] && isCycle(graph, i, visited, curPath)) // if (visited[i]) means there is no cycle starting from i
				return false;
		}
		return true;
    }
	
	
	bool isCycle(const vector<vector<int>>& graph, int src, vector<bool>& visited, vector<bool>& curPath){
		visited[src] = true;
		curPath[src] = true;
		for(int i = 0; i < graph[src].size(); i++){
			if (curPath[graph[src][i]]) return true;
			if (!visited[graph[src][i]] && isCycle(graph, graph[src][i], visited, curPath)) return true;
		}
		curPath[src] = false;// notes below
		// we don't have to revert visited[src]
		return false;
	}
};

3
[[1,0],[2,0]]

graph:
0:
1:0
2:0

visited: false, false, false
curPath: false, false, false

note that while thinking about whether it is necessary to restore curPath or not,
you should ont only consider if it works when there is a cycle, but also think about 
if it will work when there is no cycle. 

In this case, if there is a cycle we don't need to restore and the cycle can be detected;
but it might mistake a acyclic graph as cyclic like the above example.


how SWIG works? simplified wrapper interface generator

how does it connect c/c++ library to java/scala

in the swig interface file, I first confirm the scope of functions and classes I needed in the commodities analytics library to expose to java/scala, 
then included the corresponding header files in the interface file. Then i run SWIG to let it generate the cxx wrapper file which contains all the wrappers
of the functions and classes I need. Then I build the c++ project together with the generated cpp wrapper file and produce a dll for windows or .so file for linux.

As for how the dll works when we call a c++ function from java
every object is owned by c++ except for those simple primitive types like long, double, boolean.
every class has a cPtr as one of its private members
everytime a jni call is made, it has a c pointer associated with it and goes to the dll and execute the instructions located at the c pointer.


// course schedule by topological sort
class Solution {
public:
    vector<int> findOrder(int num, vector<pair<int, int>>& prerequisites) {
        // construct the edge in reverse and return the reversed topological sort directly
		vector<vector<int>> graph(num); // adjacency list
		for(int i = 0; i < prerequisites.size(); i++){
			auto edge = prerequisites[i];
			graph[edge.first].push_back(edge.second);
		}
		vector<int> topo;
		vector<bool> visited(num, false);
		vector<bool> onStack(num, false);
		for(int i = 0; i < num; i++)
			if (!visited[i]){
				if (!topologicalSort(graph, visited, onStack, topo, i))
					return vector<int>{};
			} 
		return topo;
    }
	
	bool topologicalSort(const vector<vector<int>>& graph, vector<bool>& visited, vector<bool>& onStack, vector<int>& topo, int src){
		visited[src] = true;
		onStack[src] = true;
		for(int i = 0; i < graph[src].size(); i++){
			if (!visited[graph[src][i]]){
				if (onStack[graph[src][i]] || !topologicalSort(graph, visited, onStack, topo, graph[src][i]))
					return false;
			}
		}
		topo.push_back(src);
		onStack[src] = false;
		return true;
	}
};





















