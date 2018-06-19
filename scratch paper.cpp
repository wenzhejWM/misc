
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
		ListNode* temp = res->next;
		delete res;
		return temp;
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
only one thread can have the lock

void decrement(){
	std::lock_guard<std::mutex> guard(m);
	value--;
}

std::lock_guard
mutex can only be acquired once by the same thread


std::recursive_mutex mutex;

void mul(int x){
	std::lock_guard<std::recursive_mutex> guard(mutex);
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



Two sigma:
--- construct a median heaps  
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
--- regular expression matching
	write a function to auto generate test cases
--- postfix notation calculator
--- string compression
--- Pros/cons of merge sort and quick sort
--- find all palindromes in a string
--- extend your implementation of reverse Polish notation to include more operators
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


-- Find bug in code (compute median of two sorted arrays).
-- basics, sort algorithms
-- design a fixed capacity cache with LRU eviction policy.  
-- bug tracking interview
-- system design interview questions
-- sort strings 
-- No coding on phone screen, just small question.  
-- questions on data structure, design pattern, floats and threads.  
-- how to use mac
--  Interview topics included features and syntax of Java and C++, data structures and algorithms, UNIX commands, databases, concurrency, software design.



singleton class: only one instanciation can be created.

singleton* single = singleton::getinstance();
singleton* single2 = singleton::getinstacne();

class singleton{
private:
	singleton(){}
	singleton(const singleton& other){}
	singleton& operator=(const singleton& other){}
	static singleton* obj;
	static std::mutex mt;
public:
	static singleton* getInstance(){
		mt.lock();
		if (obj == nullptr){
			obj = new singleton();
		}
		mt.unlock();
		return obj;
	}
}

is it thread safe? what happens when two or more thread are trying to create an instance of this singleton class?


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
program counter contains the address of the instruction being executed at the current time, it is increased by 1 when
each instruction gets fetched. 

register is a quickly accessible location to cpu 
threads within the same process share the same code section, data section, files and signals.

there are two types of threads; user thread and kernel thread
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
        for(int i = 1; i <= N; i++ ) ids[i] = i;
		vector<int> sz(N+1, 1);
		vector<int> res(2, 0);
		for(int i = 0; i < N; i++){
			int a = edges[i][0], b = edges[i][1];
			int r1{}, r2{};
			if (!find(ids,a,b, r1, r2)) unite(ids,sz,r1,r2);
			else res = edges[i];
		}
		return res;
    }
private:
	bool find(vector<int>& ids, int a, int b, int& r1, int& r2){
		r1 = root(ids, a);
		r2 = root(ids, b);
		return r1 == r2;
	}
	
	int root(vector<int>& ids, int i){
		while(i != ids[i]){
			ids[i] = ids[ids[i]]; // path compression by pointing to grandparent
			i = ids[i];
		}
		return i;
	}
	
	void unite(vector<int>& ids, vector<int>& sz, int a, int b){
		if (sz[a] < sz[b]){
			sz[b] += sz[a]; 
			ids[a] = ids[b];
		}
		else{
			sz[a] += sz[b];
			ids[b] = ids[a];
		}
	}
};

virtual memory
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

difference between pre and post increment(++a and a++)
int a = 0;
a++ = 0; // compiler complains, a++ not modifiable, it is an rvalue
++a = 0; // works

1. pre: do the increment to *this and return a reference to *this, lvalue, modifiable
	T& T::operator++(){ // takes no argument, return reference
		//increment
		return *this;
	}
2. post: return a copy of the object before increment, rvalue, non-modifiable
	T T::operator++(int){ // takes in a dummy int which is unused, returns a copy(temporary object)
		T copy(*this);
		++(*this); // using pre-increment
		return copy;
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

// dp with memoization
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


design patterns show relationships between classes and objects
elements of reusable object oriented software


delete an object with its base class pointer; virtual destructor

enum types can have overloaded operators 
enum color{red, green, black};
std::ostream& operator<<(std::ostream& os, const color& c){
	
}

function pointer:
int f(int a){
	
}

int (*f)(int); // this is declaring f as a pointer to a function that takes in an int and returns an int
same as declaring a function.

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
when an object is created in java, it calls the cpp constuctor in dll and returns an address(pointer) to the allocated memory
every class has a cPtr as one of its private members
everytime a jni call is made, it has a c pointer associated with it and goes to the dll and execute the instructions located at the c pointer.


// course schedule by topological sort
// it is detecting cycle at the same time
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
			if (onStack[graph[src][i]]) return false;
			if (!visited[graph[src][i]] && !topologicalSort(graph, visited, onStack, topo, graph[src][i])) return false;
		}
		topo.push_back(src);
		onStack[src] = false;
		return true;
	}
};

0: 1
1: 0

visited: false, false
onStack: false, false
topo: 

resource aquisition is initialization
memory allocation is done in constructor and deallocation is done in destructor
in this way, allocated memory is guaranteed to be held between the initialization and destruction of objects
if there is no object leak then there is no memory leak\
exception safe

copy assignment operator

MemoryBlock& operator=(const memoryBlock& other){
	if (this != &other){
		delete [] _data;
		_length = other._length;
		_data = new int[_length];
		std::copy(other._data, other._data + other._length. _data);
	}
	return *this;
}

lvalue: permanent location and it can be changed, left side of the assignment operator
rvalue: temporary object and it cannot be changed, it will disappear 
double reference: MemoryBlock&& other
other._data = nullptr
rvalue reference can detect if an object is temporary and it can make change to it
move constructor and move assignment operator 


temporary object is any non-heap object that is created but not named

C++ 11 uniform initialization with {}

A(const A& other) = delete; // copy constructor is not allowed 

Summary of C++ 11 new features:
-- lambda expression
-- auto, type deduction
-- uniform initialization with curly braces
-- STL algorithms
-- threading library, #include <thread>
-- deleted and defaulted functions
-- nullptr
-- delegating constructors, i.e., calling another constructor inside another constructor
-- smart pointers
-- rvalue reference, able to change the value of an rvalue #include <unitily>


std::swap with move semantics

template<class T> 
void swap(T& t1, T& t2){
	T temp = std::move(t1);
	t1 = std::move(t2);
	t2 = std::move(temp);
}

no new resource allocated in the process and nothing is copied


C++ memory layout:
1. stack: local variables, arguments passed to function calls, return addresses at function calls
2. heap: dynamically allocatd memory
3. data: globals, static variables, literals?
4. code section: holds the compiled code of the program

Anatomy of a thread:
-- threads share code and data memory segment 
-- each thread has its own stack allocated in the stack segment of the process' address space
-- the default size of a thread's stack is assigned by the system



 c++ compiling process:
 1. preprocessing: deals with #include and #define, the output is a pure c++ file without , #if, #ifdef, #ifndef
	#error
 2. compiler generates assembly code, assembler then assembles that code into machine code producing the actual binary file
	each cpp will be converted into an object file(binary form of compiled code)
	These object files can be put in static libraries.
	Object files can refer to symbols that are not defined, i.e., declaration without implementation is fine
 3. linking: produces a shared (or dynamic) library or an executable
 
for template classes the declaration and the implementation should be in the same header file
suppose I have a temp.h which contains a class declaration 
and temp.cpp which contains the implementation
and in another cpp file, func.cpp, we have temp<int> tint and #include "temp.h"
the preprocessing will only copy temp.h(declaration) into func.cpp
and also the compiler will not generate code for template class implementation unlesss it is instantiated.
Workaround: if we want to separate decl and impl, we must explicitly instantiate template class inside the impl file
so that the impl will be accessible when the compiler generates the code for temp<int>
 


overloading operator ->, this one is tricky
the return type has to be either a pointer that can be dereferenced,
or reference to an object that has a member pointer that can be dereferenced.
you don't have to call p->->func() in either case, just p->func()

plain old data: no virtual functions, no self defined constructors or destructors
it is exactly its data members

in unique_ptr, there is no copy constructor or assignment operator
similar to auto-ptr which has been deprecated because it does not support move semantics

class MedianFinder {
private:
    priority_queue<int> maxHeap;
	priority_queue<int, vector<int>, greater<int>> minHeap;
public:
    /** initialize your data structure here. */
    MedianFinder() {
        
    }
    // maxHeap mid minHeap
    void addNum(int num) {
		if (maxHeap.size() == 0) maxHeap.push(num);
        if (num <= maxHeap.top()){
			if (maxHeap.size() <= minHeap.size()) maxHeap.push(num);
			else{
				minHeap.push(maxHeap.top());
				maxHeap.pop();
				maxHeap.push(num);
			}
		}
		else{
			if (maxHeap.size() >= minHeap.size()) minHeap.push(num);
			else{
				maxHeap.push(minHeap.top());
				minHeap.pop();
				min.heap.push(num);
			}
		}
    }
    
    double findMedian() {
		int smax = maxHeap.size(), smin = minHeap.size();
        if (!smax && !smin) return 0;
		if (!smin()) return maxHeap.top();
		if (smax == smin) return (maxHeap.top() + minHeap.top())/2.0;
		if (smax < smin) return minHeap.top();
		return maxHeap.top();
    }
};


class Solution {
public:
    bool areSentencesSimilar(vector<string>& words1, vector<string>& words2, vector<pair<string, string>> pairs) {
        int n1 = words1.size(), n2 = words2.size();
        if (n1 != n2) return false;
		unordered_map<string, string> map1, map2;
		for(auto p: pairs){
			map1[p.first] = p.second;
			map2[p.second] = p.first;
		}
		for(int i = 0; i < n1; i++){
			if (map1[words1[i]] == words2[i] || map2[words2[i]] == words1[i]) continue;
			return false;
		}
		return true;
    }
};

Rule of Five:
self-defined constructor // construct from arguments
copy constructor // construct from another existing object, two objects in total
copy assignment operator= //  construct from another existing object, two objects in total
move constructor
move assignemnt operator=

// big five of the following class
class DirectorySearchResult {
public:
  DirectorySearchResult(
    std::vector<std::string> const& files,
    size_t attributes,
    SearchQuery const* query)
    : files(files),
      attributes(attributes),
      query(new SearchQuery(*query))
  { }

  ~DirectorySearchResult() { delete query; }

private:
  std::vector<std::string> files;
  size_t attributes;
  SearchQuery* query;
};

// copy constructor
DirectorySearchResult::DirectorySearchResult(const DirectorySearchResult& other){
	files = other.files;
	attributes = other.attributes;
	query = other.query;
}

// copy assignment: delete the old content in this and copy the new content into it
DirectorySearchResult& DirectorySearchResult::Operator=(const DirectorySearchResult& other){
	if (this == &other) return *this; // assigning itself
	files = other.files;
	attributes = other.attributes;
	delete query;
	query = other.query;
	return *this;
}

// move constructor
DirectorySearchResult::DirectorySearchResult(DirectorySearchResult&& other){//rvalue reference
	files = std::move(other.files);
	attributes = other.attributes;
	query = other.query;
	other.query = nullptr;  // move other.query
}

// move assignment operator
DirectorySearchResult& DirectorySearchResult::operator=(DirectorySearchResult&& other){
	if (this == &other) return *this;
	files = std::move(other.files);
	attributes = other.attributes;
	query = other.query;
	other.query = nullptr;
	return *this;
}

=================
auto p = make_unique<int>(1); // no new used

Exception-Safety of function calls

consider this function signature:
void f(std::unique_ptr<T1>, std::unique_ptr<T2>);

and a function call:
f(std::unique_ptr<T1>(new T1), std::unique_ptr<T2>(new T2));

This is not exception safe. 
The following is a possible order of argument evaluation of this call:
1. allocate memory for T1
2. construct object of type T1
3. allocate memory for T2
4. construct object of type T2
5. construct unique_ptr<T1>
6. construct unique_ptr<T2>

if an exception is thrown in T2's constructor, T1 memory leak
The problem is at interleaving of evaluation of expressions of function call arguments

make_unique<T>() to save:
it is a function inside which the unique_ptr constructor is called, so it cannot be interleaved.


https://softwareengineering.stackexchange.com/questions/183723/low-latency-unix-linux

--- lockfree programming library, why is it better than locks
--- beat VWAP, TWAP
--- value at risk, CVAR
--- ponse scheme
--- trading protocols, fix engine/protocol
--- tune linux for low latency
--- functional programming to improve concurrency and scalability
--- moore's law
--- scalable coce to run on multiple cores
--- focus on Sockets, RSS feeds (double check that one), system security (IMHO either sand boxing or a ROM OS for something like this), 
encryption and authentication and the protocols specific to the exchange your choose. 

in linux kernel, context switching involves switching registers, stack pointer, program counter...
e.g. accessing disk, wait

Steps of context switching:
1. store all the states of the current process: registers, stack pointer, program counter
   process control block
2. a handle to the PCB is added into a queue of processes that are ready to run: ready queue
3. the operating system chooses a process from the ready queue, according to priority or other criterion

cost of context switching:
--- running task scheduler
--- TLB flushes
--- indirect: sharing CPU cache between multiple tasks.
--- switching between threads of a single process is faster than two processes because threads share memory, so TLB flush is not necessary
--- quantlib
--- unit test for multithreaded application
--- thread safe with stl

reducing the number of threads can reduce overhead of context switching
track the context switching per second

real time application/system generally employ premptive scheduling to guarantee ciritical threads are executed immediately
preemptive scheduling:
if a process is running and another process with higher priority is ready to run, the scheduler does a context switching

thread starvation: a thread is always waiting and other threads with higher priorities are consuming a very long time.


--- multiple locks or single lock for an object?
more locks means more parallelism, more overhead, more difficult to debug

--- how efficient is it to use a mutex?
a mutex has two major parts: (1) a flag indicating whether it is locked
(2) a waiting queue of threads

locking an unlocked mutex is cheap: just turn on the flag, without system call
acquiring locked mutex makes system call and add the thread into the waiting queue
unlocking is cheap if the waiting queue is empty, otherwise it has to wake one of the threads in the queue

--- system call: a program request service from the kernel of the operating system, it is the interface between process and operating system

-- attempting to lock an already locked mutex is called contention, it should be minimized

---VWAP volume weighted average price
it reduces the noise in floating prices
institutional investors try to buy below and short term investors opposite
it is more responsive to price moves at the beginning of the day, more useful for short term
less responsive at the end of day because of large accumulated volume, more useful for institutional

--- implementation shortfall:
difference in price between a theoretical portfolio and the implemented portfolio

--- three steps of trading: 
signal generation
trade execution: happens after the trading signal has triggered a buy or sell. it determines how the order is structured: position size and limit levels
post trade performance analysis


--- bool in cpp has a size of 1 byte: this is because the unit of addressing is 1 byte
a pointer cannot point to a bit

--- implied vol can be calculated by bisection method: because option price is strictly increasing in vol
--- Newton method is the best

#if 0
	code prevented from compiling
#endif

#ifndef 
	cerr<<"error information\n";
#endif

--- predefined macros:
__LINE__: the current line number when it is being compiled
__FILE__: the current file name
__DATE__: the date of the translation of the source file into object code
__TIME__: hour:minute:second when being compiled


--- #pragma comment(lib, libname) // it leaves a comment in the generated object code to tell the linker to add this library into its dependencies.
// equivalent to add it into the project properties at Linker->Input->Additional

--- #define IDATA_EXPORT __declspec(dllexport)
class IDATA_EXPORT IData // macro before class name: this class will be exported if compiled to dll by configuration

--- set bit on given positions
x = (1 << n1) | (1 << n2) | ...
return (n | x);

--- template explicit instantiation
template<class T>
void func(){....}
// explicit instantiation:
template<> void func<int>(); 

--- anonymous namespace
namespace{
	void func(){std::cout<<"in anonymous namespace\n";}
}

func can be used in this file without qualifiers, this is because the compiler generates a unique name for this namespace
compiler does this:
namespace unique{}
using namespace unique; // this is why we can use func directly in this file
namespace unique{
	......
}

--- separate parts of the same namespace can be spread over multiple files
--- nested namepsace 
namespace AAA{
	void func(){...}
	namespace BBB{
		void func(){
			::func(); // calling root namespace 
		}
	}
}
--- std::cout: "cout" is a qualified name, "std" is a qualifier
---- about typename
template<class T> or template<typename T>

template<class T>
class AAA{
	typename T::iterator * iter; // typename is required before dependent type, otherwise it is just multiplication
};

iter's type is dependent on T and its assuming T has a class called iterator.
typedef typename T::iterator Tit;

--- bitset
it can convert positive/negative to 01 string
but can only convert 01 string to unsigned long(long) only

auto num1 = bitset<32>(-124343);
auto num2 = bitset<32>("1010010101....");
string bin = num1.to_string();
unsigned long temp = num2.to_ulong();

--- recursion versus iteration
recursion is usually slower in C, java, python and other imperative languages because it wastes space for stack frame
but it is faster in some functional programming languages

--- exceptions 
try{
	...
}catch(ExceptionType1 e){
	...
}catch(ExceptionType2 e){
	...
}catch(...){/// it cathces all possible exceptions
	
}

--- const char* c_str() noexcept{} // convert string to c string
	string constructor converts c string to string
--- return by address usaully happens when the returning varialbe is on heap

--- prevent memory leak from a programmer's view:
1. reallocate memory to the same pointer only the old one is deleted
	char* str = new char[30];
	str = new char[20]; // 30 chars are leaked 
2. watch out for pointer assignments, every chunk of memory in heap has to be associated with a pointer
	char* p1 = new char[30];
	char* p2 = new char[50];
	p1 = p2; // char[30] is leaked
3. be careful with local pointers pointing to heap memory
	void leak(int x){
		char* p = new char[x];
		// delete [] p; // if forgot to delete, p will be destroyed when out of scope of leak
	}
4. pay attention to "[]"
	int* p = new int[20];
	//delete p; // it only deletes a single int
	delete [] p; 

--- exception class hierarchy allows for polymorphic exceptions 
	runtime_error contains: overflow_error, range_error, system_error, underflow_error
	logic_error contains: domain_error, future_error, invalid_argument, length_error, out_of_range
	...

---  const values and references have to be initialized in the initialization list of the constructor
--- external sorting algorithms:
type1: distribution sort, like quick sort
type2: external merge sort

type1: 


type2: 



--- two goals of Feds:
1. keep inflation at about 2% per year
2. minimizing unemployment
trade-off between full employment and inflation
if the economy goes too fast, any excess supply of essential resources begin to run out, the scarcity makes prices rise, leading to inflation.
philips curve: full employment and inflation



gradient descent in c++
c++ profiler

distributed hash table
volatile related to multi-threading
multithreading is easier in functional programming

========================================================================= CPP 11 ========================================
--- for plain old data, a move is the same as a copy
--- constexpr is to improve performance by computing at compile time instead of runtime
	once a program is compiled......
	in C++ 11 constexpr should have only one return statement
	constexpr functions can only call other constexpr functions
	it should not be void type 
--- std::initializer_list<int>
--- static assertions, evaluated at compile time
	constexpr int x = 0;
	constexpr int y = 1;
	std::static_assert(x == y, "x != y")
--- auto variables are deduced by the compiler according to initilizer type
	autp g = new auto(1);
--- [a,&b]() -> int&{return x;}
	[x]()mutable{x=2;}
--- decltype(b) a = b;
	template<class X, class Y>
	auto add(X x, Y y) -> decltype(x+y){
		return x + y;
	}
--- tmeplate alias 
	template<typename T>
	using Vec = std::vector<T>;
	Vec<int> vint;
	
	using String = std::string;
	String s{"foo"};
--- constexpr 
	constant expressions evaluated at compile time
--- A() = default;  // all the members of class A will be default initialized
--- to_string(123.123);
--- autp profile = std::make_tuple(51, "wenzhe", 120000);
	auto num = std::get<0>(profile);
	auto name = std::get<1>(profile);
	auto salary = std::get<2>(profile);
	std::tie(num, name, std::ignore) = std::make_tuple(51, "wenzhe", 120000); // unpacking tuple
--- std::make_shared<A>(A(1,2));
	prevent using new
	only type once as opposed to std::shared_ptr<A>(new A(1,2));
	exception safety

	
========================================================================= CPP 14 ========================================
--- binary literals
	0b110  // 6
	0b1111'1111 // 255, use ' to separate digits
--- generic lambda expression, enables polymorphic lambdas
	auto identity = [](auto x){return x;};
	int three = identity(3); // 3
	std::string foo = identity("foo");  // foo
--- lambda capture initializers
	auto gen = [x=0](){return x;};
	auto gen = [x=0]() mutable{return ++x;};
	auto a = gen(); // 1
	auto b = gen(); // 2
	auto c = gen(); // 3
	
	auto p = std::make_unique<int>(1);
	auto task1 = [=](){*p = 5;}; // does not compile, capture p by copy which is not allowed.
	auto task2 = [p = std::move(p)](){*p = 5;};  // now it's fine, the original p is empty
	
--- return type deduction
	auto f(int i){return i;}   // deduce return type as int
	template <class T>
	auto& f(T& t){return t;}
	
	auto g = [](auto& x) -> auto& {return f(x);};   // very generic lambda expression
--- std::make_unique
--- constexpr can have multiple return statement

========================================================================= CPP 17 ========================================
--- template argument deduction for class templates
	tmeplate<class T = float>
	struct MyContainer{
		T val;
		MyContainer():val(){}
		MyContainer(T v): val(v){}
	};
	MyContainer c1{1}; // don't need MyContainer<int> 1, it is deducted
	MyContainer c2; // 

--- inline variables
	struct S{int x;};
	inline S x1 = S{123};  // copy the constructor code here
	S x2 = S{321};
	
--- nested namepace 
	namespace A{
		namespace B{
			namespace C{
				int i;
			}
		}
	}
	
	namespace A::B::C{
		int j;
	} 

--- std::optional
--- std::any
	a type-safe container for single values of any type
	std::any x{5};
	x.has_value(); // true
--- std::apply(add, std::make_tuple(1,2));
	
============================================== STL ==================================
--- queue
	a queue should have at least these operations: empty, size, front, back, push_back, pop_front
	
	T& front(); // same for back
	const T& front() const;
	
	void push(T& val);
	void push(const T& val);
	
	void pop();  // calls the removed element's destructor
	
	template<class... Args>
	void emplace(Args&&... args);
	
	void swap(queue& x) noexcept;  // exchange the contents of the container 
	e.g., q.swap(p);
	
--- stack: empty, size, top, push, pop, emplace, swap

--- map: 
	tmeplate<class Key, class V, class Compare = less<Key>, class Alloc = allocator<pair<const Key, V>>>
	class map;
	
	typedef std::pair<Key, T> value_type;
	std::pair<iterator, bool> insert(const value_type& val);  // 1 if new, 0 otherwise
	template <class InputIterator>
	void insert(InputIterator first, InputIterator second);
	
	size_type erase(const key_type& k); // returns the number of elements removed
	iterator erase(const_iterator position); 
	iterator erase(const_iterator first, const_iterator last); // returns the iterator following the last element removed or map::end
	
	void clear() noexcept;  // clear all the elements and leaves the size as 0
	
	iterator find(const key_type& k); // returns the iterator to k, otherwise map::end
	size_type count (const key_type& k) const;
	
	iterator lower_bound(const key_type& k); // iterator to the first element that equal or goes after
	iterator upper_bound(const key_type& k); // iterator to the first element that goes after
	
	it++, it--  // both work
	
--- set: similar to map

--- vector:
	template<class T, class Alloc = allocator<T>> class vector;
	the underlying dynamic array may need to be reallocated in order to grow in size
	void resize(size_type n); 
	void resize(size_type n, const value_type& val);
	1. n smaller than size, remove those beyond and detroy
	2. n greater than size, existing elements unaffected
	3. n greater than capacity, reallocate 
	
	void reserve(size_type n); 
	if n is smaller than capacity, nothing happens
	
	reference back();
	const_reference back() const; // returns a reference to the last element
	// an empty container cannot call this // undefined behavior
	
	iterator insert(iterator position, ....);//  insert right BEFORE the specified position
	
	std::vector<int> myvector(3, 100);
	std::vector<int>::iterator it;

	it = myvector.begin(); cout << *it << endl;
	it = myvector.insert(it, 200); cout << *it << endl;

	myvector.insert(it, 2, 300); cout << myvector[0] << endl; // now "it" has been invalidated because reallocation happened
	
	iterator erase(const_iterator position); return the iterator to the element following the last erased one
	
--- unordered_map
	template<class Key,
			class T,
			class Hash = hash<key>,
			class Pred = equal_to<Key>,
			class Alloc = allocator<pair<const Key, T>>
			> class unordered_map;
	
	iterator find(const Key& key);  // return the iterator to the found element or std::unordered_map::end
	
	size_type bucket_count() const noexcept; // returns the number of buckets in the unordered_map container
	// every time the bucket_count is increased(lowers the load factor), rehash happens
	size_type bucket(const key_type& k) const; // return the bucket number where the element with key k is located
	float load_factor() const noexcept; // size / bucket_count
	float max_load_factor() const noexcept; // return the threshold for rehash, default value is 1.0
	void max_load_factor(float z); // set the max_load_factor to z
	size_type bucket_size(sie_type n) const; // return the number of elements in bucket n
	// int s = bucket_size(bucket(k));
	
--- the concept of a reference is not tied to any specific implementation. depends on the compiler
	sometimes the compliler chooses to implement this by using a pointer. 
	But often, it implements it by doing nothing at all, just generate code which refers directly to the original object
	
--- unique_ptr
	template< class T, class D = default_delete<T>> class unique_ptr;
	no copy constructor
	no copy assignment(but it has move assignment)
	* and -> work but no pinter arithmetic
	
	{
		auto deleter = [](int* p){
			delete[] p;
			cout<<"array deleted\n";
		}
	
		unique_ptr<int, decltype(deleter)> up(new int, deleter);   // notice the usage of decltype and construction from raw pointer
	}
	
	pointer get() const noexcept; // returns (but not release) the stored pointer, cannot be used to construct another managed pointer
	pointer release() noexcept; // return the stored pointer and the stored becomes nullptr 
	void reset(pointer p = nullptr) noexcept; // destroy the managed object and takes ownership of p
	element_type& operator[](size_t i) const; // only defined in array-specialization
	unique_ptr<int[]> foo(new int[4]);  // cannot be unique_ptr<int>
	foo[3] = 2;
	
--- std::make_unique
	
--- std::make_shared
	template<class T, class... Args>
	shared_ptr<T> make_shared(Args... args);

--- shared_ptr
	template <class T>
	class shared_ptrp;
	shared_ptr objects can only share onwership by copying their value, not by construction from the same raw pointer
	
	reset, get
	
	long int use_count() const noexcept; // returns the number of shared_ptr that share the ownership over the same pointer 
	
--- priority_queue
	template<class T, class Container = vector<T>, class Pred = less<T>>
	class priority_queue;
	
	empty, size, pop
	
	const_reference top() const; // note const reference

--- deadlock prevention
	lock ordering, only when you can decide the order of locks
	lock timeout: if a thread does not succeed in taking the necessary locks within the given time, it will free all the obtained locks, wait and then try again.
	deadlock detection with graph which stores thread and corresponding lock information, once detected,release all the locks and wait and retry, or release some locks by priority

	
--- starvation
--- graph BFS traversal
--- CPU cache
	when the processor needs to read from or write to a location in main memory, it first checks whether a copy of htat dat is in the cache. 
	data is transferred between memory and cache in blocks of fixed size, cache block
	every core has L1 and L2 cache
	L1 is split into data cache and instruction cache
	L2 is usually not split
	L3 cache is shared between cores
--- cache line
	
--- memory alignment
	
	
	
	
	
--- LRU cache implementation
--- a word is a unit of data used by CPU, usually 8, 16, ... 64
--- void* memset(void* ptr, int value, size_t num);
	set num bytes of the block of memory pointed by ptr to value(interpreted as unsigned char)

--- virtual inheritance: all derived class from the base share a common subobject

--- c++14 supports runtime_sized arrays 
--- throwing an exception is at runtime only
--- b1 && b2 // if b1 is false b2 will not be evaluated, this is called short circuiting

--- inline recursive function
	doable, but compiler does not know the depth of recursion at compile time
	it might generate a few (fixed depth) and the rest is still recursive
--- exception and stack unwinding
--- #include <algorithm> works on range of elements, array or stl containers, never affects the size or storage allocation of the container
	all_of, any_of, none_of
	Function for_each(first, last, function); // for_each(nums.begin(), nums.end(), [](int i){return i * 2;}); returns function
	find(first, last, v); // if not found, return last
	InputIterator find_if (InputIterator first, InputIterator last, UnaryPredicate pred);
	min, max, minmax can be used with initializer list: min({1,2,43,6,3,3})
	min_element, max_element(nums.begin(), nume.end());

--- 5 types of iterators:(low to high hierarchy)
	input iter: equality/inequality; dereferencing; ++ only; only accessing, no assigning 
	output iter: cannot be compared; deref; ++; only assigning, no accessing(which means if you access it, it must be assigned)
	forward: combination of input and output iterator; support equality/inequality comparison; only ++; no < or >; no arithmetics; 
	bidirectional:(list, map, set) ++ and --; no relational; no arithmetics; 
	random access:(supported only in deque and vector) no relational; arithmetics supported; offset allowed" a[3]"
	
--- unordered_map supports only forward iterator
--- functional
	greater<int>// this one is a class
	greater<int>() // this one is like a lambda expression
	bit_and, bit_or, bit_xor, minus, negate, plus, ......


--- std::function

--- commodities CPPUnit
--- stdout
--- std::cerr
--- std::cout is buffered: information sent to cout does not appear on the screen until it is flushed
	std::endl will flush it: cout<<...<<endl;
	cout.flush();
	cout<<...<<flush;
	Read from the cin stream or write to the cerr or clog streams. Because these objects share the buffer with cout, each flushes the contents of the buffer before making any changes to it.
	
	
--- std::thread my_thread(background_task());
	// this is not the right way to instantiate a thread object
	//in C++, everything that can be treated as a function will be treated as a function
	// this is declaring function my_thread that takes a function pointer and returns std::thread
	// the function pointer takes no argument and returns a background_task object
	
	// the correct way to do this is:
	std::thread my_thread{background_task()};
	std::thread my_thread((background_task()));
	
--- std::thread destructor calls std::terminate()
--- if a pointer or reference to a local variable is passed into a thread, the local variable might be destroyed before the thread finish.
	the solution is to make the thread self-contained and copy the data into the thread
	Be wary of objects that contains a pointer or reference to local variables
--- a detached thread's ownership is passed to C++ runtime library and it becomes a daemon thread
--- an initialized thread is joinable and has its own unique thread id
	an uninitialized thread is unjoinable and all such threads have the same thread id
	
	
--- a functor object passed into thread instantiation is COPIED 
--- heap checking in linux: These debugging features are controlled through an environment variable, MALLOC_CHECK_. When this is set to 1, the error is reported on stderr. When set to 2, any error causes the program to abort with a core dump being generated if possible. It is normally possible to use a debugger to analyse the core dump and trace back to the error.
	export MALLOC_CHECK_ = 1
	
--- misaligned exception

--- oversubscription: creating more threads than the hardware can support, e.g., 100 threads with only 32 cores
	context switching decreases performance



--- std::enable_if<bool Cond, class T>::type
	if Cond is true, return T; // T by default is void
	
	typename std::enable_if<std::is_integral<T>::value, bool> is_odd(T i){return (i&1 == 1);}
	// if T is not integral type, it will not compile
	
	
--- std::thread::hardware_concurrency() // this returns an indication of the number of threads that can truly run in parallel
	
--- mutex and pointer: when a function that is protected by a mutex returns a pointer then it is not safe.
1. what editors do you use under linux?
	practice vim
2. how do you compile a project?
	practice make file, gmake, premake
3. compiler?
	often used compile options of gcc, versions
4. MS c++ code of conduct
5. MS core C++, dave handley
6. valgrind, cachegrind, gdb debugger, break points, monitoring variables
7. jumping around files 
8. vim plugins 
	
--- a library is binary compatible, if a program linked dynamically to a former version of the library continues to run with newer versions of the library without the need to recompile.
--- inline with definition
	
============================= coding standards ============================
--- includes should be listed in groups, going from specific to standard (std..)
--- use forward declaration whenever possible
--- inside a class: public before protected before private
--- 


日主： 癸水
印： 金 生我  智慧  福气  父母 长辈 同：偏印  异 正印
官杀： 土 克我  压力 官： 异性 保守  杀：同性  冲劲   同： 偏官  七杀  异 正官
食伤：木  我生之物   食神：同   伤官：异
财： 火 我克      正财： 异    偏财：  同 
比劫： 水  同我   比：同   劫： 异
	
	
	
	
	
	
	
	
	
	
	
	
	
	