//Vectors, part of C++ Standard Template Library (STL), can dynamically resize themselves
//as compared to: Arrays, which have a fixed size that must be determined at compile time

#include <iostream>
#include <vector>    
#include <stdexcept> 
#include <print> // C++23 header for std::print and std::println

void outputVector(const std::vector<int>& items); // display the vector which can’t be modified due to “const”
void inputVector(std::vector<int>& items); // input values into the vector which can be modified due to “&”

int main() {
   //note that if {}, instead of (), used below,
   //then it would have been for 1 element initialized to 7 and 10 respectively
   std::vector<int> integers1(7); // 7-element vector<int> initialized to 0s
   std::vector<int> integers2(10); // 10-element vector<int> initialized to 0s

   // print integers1 size and contents
   std::cout << "Size of vector integers1 is " << integers1.size()
      << "\nvector after initialization: ";
   outputVector(integers1);

   // print integers2 size and contents
   std::cout << "\nSize of vector integers2 is " << integers2.size()
      << "\nvector after initialization: ";
   outputVector(integers2);

   // input and print integers1 and integers2
   std::cout << "\nEnter 17 integers:\n";
   inputVector(integers1);
   inputVector(integers2);

   std::cout << "\nAfter input, the vectors contain:\n"
      << "integers1: ";
   outputVector(integers1);
   std::cout << "integers2: ";
   outputVector(integers2);

   // use inequality (!=) operator with vector objects
   std::cout << "\nEvaluating: integers1 != integers2\n";

   if (integers1 != integers2) {
      std::cout << "integers1 and integers2 are not equal\n";
   }

   // create vector integers3 using integers1 as an     
   // initializer; print size and contents              
   std::vector integers3{integers1}; // copy constructor

   std::cout << "\nSize of vector integers3 is " << integers3.size()
      << "\nvector after initialization [after std::vector integers3{integers1}]: ";
   outputVector(integers3);

   // use overloaded assignment (=) operator              
   std::cout << "\nAssigning integers2 to integers1 [by simple assignment: integers1 = integers2]:\n";
   integers1 = integers2; // assign integers2 to integers1

   std::cout << "integers1: ";
   outputVector(integers1);
   std::cout << "integers2: ";
   outputVector(integers2);

   // use equality (==) operator with vector objects
   std::cout << "\nEvaluating [needs to be same data-type, same length, and equal corresponding elements]: integers1 == integers2\n";

   if (integers1 == integers2) {
      std::cout << "integers1 and integers2 are equal\n";
   }

   // use the value at location 5 as an rvalue
   std::cout << "\nintegers1.at(5) is " << integers1.at(5);

   // use integers1.at(5) as an lvalue
   std::cout << "\n\nAssigning 1000 to integers1.at(5)\n";
   integers1.at(5) = 1000;
   std::cout << "integers1: ";
   outputVector(integers1);

   // attempt to use out-of-range index                   
   try {
      std::cout << "\nAttempt to display integers1.at(15)\n";
      std::cout << integers1.at(15) << '\n'; // ERROR: out of range
                                // .. hence, skipping rest of try block/statement, and going to catch handler below
      std::cout << "this will NOT get printed\n" << '\n'; // due to ERROR above, this will be skipped
   }
   catch (const std::out_of_range& ex) {
      std::cerr << "An exception occurred: " << ex.what() << '\n';
   }

   // changing the size of a vector
   std::cout << "\nCurrent integers3 size is: " << integers3.size();
   integers3.push_back(1000); // add 1000 to the end of the vector
   std::cout << "\nNew integers3 size is [after integers3.push_back(1000)]: " << integers3.size() << "\nintegers3 now contains: ";
   outputVector(integers3);

   // changing the size of a vector, again
/*
Note:
std::vector does not have a pop_front() method. This is because std::vector is designed for efficient access and modification at the back of the sequence. Removing an element from the front of a vector requires shifting all subsequent elements, which is an O(n) operation.
If you need to efficiently remove elements from the front of a sequence, consider using std::deque or std::list, which are designed for such operations.
*/
   std::cout << "\nCurrent integers3 size is: " << integers3.size();
   if (!integers3.empty()) {
      integers3.erase(integers3.begin()); // Removes the first element
   }   
   std::cout << "\nNew integers3 size is [after integers3.erase(integers3.begin()); std::vector does not have a pop_front() method]: " << integers3.size() << "\nintegers3 now contains: ";
   outputVector(integers3);

   std::println("\nTHE END");
   return 0;
}

// output vector contents
void outputVector(const std::vector<int>& items) {
   for (const int& item : items) {
      std::cout << item << ' ';
   }
   std::cout << '\n';
}

// input vector contents
void inputVector(std::vector<int>& items) {
   for (int& item : items) {
      std::cin >> item;
   }
}

/*
Size of vector integers1 is 7
vector after initialization: 0 0 0 0 0 0 0

Size of vector integers2 is 10
vector after initialization: 0 0 0 0 0 0 0 0 0 0

Enter 17 integers:
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17

After input, the vectors contain:
integers1: 1 2 3 4 5 6 7
integers2: 8 9 10 11 12 13 14 15 16 17

Evaluating: integers1 != integers2
integers1 and integers2 are not equal

Size of vector integers3 is 7
vector after initialization [after std::vector integers3{integers1}]: 1 2 3 4 5 6 7

Assigning integers2 to integers1 [by simple assignment: integers1 = integers2]:
integers1: 8 9 10 11 12 13 14 15 16 17
integers2: 8 9 10 11 12 13 14 15 16 17

Evaluating [needs to be same data-type, same length, and equal corresponding elements]: integers1 == integers2
integers1 and integers2 are equal

integers1.at(5) is 13

Assigning 1000 to integers1.at(5)
integers1: 8 9 10 11 12 1000 14 15 16 17

Attempt to display integers1.at(15)
An exception occurred: invalid vector subscript

Current integers3 size is: 7
New integers3 size is [after integers3.push_back(1000)]: 8
integers3 now contains: 1 2 3 4 5 6 7 1000

Current integers3 size is: 8
New integers3 size is [after integers3.erase(integers3.begin()); std::vector does not have a pop_front() method]: 7
integers3 now contains: 2 3 4 5 6 7 1000

THE END
*/
