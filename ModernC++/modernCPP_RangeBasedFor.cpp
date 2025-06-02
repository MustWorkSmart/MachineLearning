//range-based for with initialization, std::array

/*
std::array in C++ offers several advantages over using raw C-style arrays (declared with brackets []). 
These advantages primarily revolve around safety, convenience, and better integration with modern C++ practices. 

Safety:
•	Bounds Checking:
std::array can provide bounds checking (using the .at() method) which can prevent buffer overflows. Traditional arrays do not have this built-in protection, potentially leading to undefined behavior and security vulnerabilities.
•	No Pointer Decay:
C-style arrays automatically decay into pointers when passed to functions, losing size information. std::array does not decay to a pointer, preserving its size and type information.

Convenience:
•	Value Semantics:
std::array behaves like a regular object, meaning it can be copied by value and assigned without issues. C-style arrays require manual memory management when copying or assigning, often leading to errors.
•	Size Information:
std::array stores its size as a member, which can be accessed using the .size() method. C-style arrays require manually tracking the size, which is prone to errors.
•	Iterator Support:
std::array provides iterators, making it compatible with standard algorithms and range-based for loops. This simplifies working with the array and improves code readability.

Better Integration with Modern C++:
•	Standard Library Compatibility:
std::array is a standard container, making it consistent with other C++ data structures. It can be used with various standard algorithms and functions.
•	Type Safety:
std::array is a template, which provides better type safety. The size of the array is part of its type, allowing the compiler to detect size mismatches at compile time.

Performance:
•	std::array has no performance overhead compared to C-style arrays. Both have the same performance characteristics in terms of memory access and storage.
*/

#include <format>
#include <iostream>
#include <array>
#include <print> // C++23 header for std::print and std::println

int main() {
   std::array items{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}; // type inferred as array<int, 10>

   std::println("Using: range-based for (with or without initialization), and std::array ...\n");

   // display items before modification
   std::cout << "items before modification [for (const int& item : items)]: ";
   for (const int& item : items) { // item is a reference to a const int – NOT modifying it!
      std::cout << std::format("{} ", item); 
   }                       
 
   // multiply the elements of items by 2
   for (int& item : items) { // item is a reference to an int – modifying!
      item *= 2;
   }

   // display items after modification
   std::cout << "\nitems after modification [for (int& item : items)]: ";
   for (const int& item : items) {
      std::cout << std::format("{} ", item);
   }

   // sum elements of items using range-based for with initialization
   std::cout << "\n\ncalculating a running total of items' values [for (int runningTotal{0}; const int& item : items)]:\n";
   for (int runningTotal{0}; const int& item : items) {
      runningTotal += item;
      std::cout << std::format("item: {}; running total: {}\n", item, runningTotal);
   }

   std::println("\nTHE END");
   return 0;
}