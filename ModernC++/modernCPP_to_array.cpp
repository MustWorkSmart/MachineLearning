/*
Background info:
int* - Pointer to int
int const * - Pointer to const int
int * const - Const pointer to int
int const * const - Const pointer to const int

const int * and int const * are equivalent in C and C++. They both declare a pointer to a constant integer. This means that the value the pointer points to cannot be modified through that pointer. However, the pointer itself can be changed to point to a different memory location.
The const keyword applies to the thing immediately to its left. If there's nothing to the left, it applies to the thing to its right. In both const int * and int const *, the const applies to the int, indicating that the integer being pointed to is constant. 
.. hence, “const int * const” and “int const * const” are the same as well

For example:
int x = 5;
const int *ptr1 = &x; // pointer to const int
int const *ptr2 = &x; // pointer to const int

x = 10; // OK, x itself is not const
// *ptr1 = 20; // Error: cannot modify the value through ptr1
// *ptr2 = 30; // Error: cannot modify the value through ptr2

int y = 15;
ptr1 = &y; // OK, ptr1 can point to a different address
ptr2 = &y; // OK, ptr2 can point to a different address

In this example, both ptr1 and ptr2 are pointers to a constant integer. 
The value of x can be changed directly, but not through ptr1 or ptr2. 
The pointers themselves can be reassigned to point to y.
*/

/*
The C++20 std::to_array function in C++ is a utility that converts a C-style array or a braced-initializer list into a std::array. Here are some of its benefits:
•	Type Deduction:
std::to_array automatically deduces the element type and size of the array, reducing boilerplate code and potential errors.
•	Value Semantics:
std::array has value semantics, meaning it can be copied and assigned without issues of array decay, unlike C-style arrays.
•	STL Compatibility:
std::array is a standard container, making it compatible with STL algorithms and iterators.
•	No Array Decay:
C-style arrays decay to pointers when passed to functions, losing size information. std::to_array avoids this issue.
•	Safety:
std::array provides bounds checking, preventing common errors like accessing out-of-bounds memory.
•	Return from Functions:
std::array can be returned from functions by value, which is not possible with C-style arrays.
•	Built-in Comparisons:
std::array supports built-in lexicographical comparisons through the standard comparison operators (like ==, !=, <, <=, >, >=). These operators compare the contents of two std::array objects element by element. 
•	Copy Construction:
std::array can be copy-constructed and copy-assigned.
•	Conciseness:
Using std::to_array can sometimes lead to more succinct code, particularly when initializing arrays.
•	No Heap Allocation:
std::array allocates memory on the stack, avoiding the overhead of dynamic memory allocation, unlike std::vector.
•	Interoperability:
It facilitates the conversion of C-style arrays to C++ arrays, which can be beneficial when interacting with C libraries.
While std::to_array offers many benefits, it is important to consider that std::array has a fixed size determined at compile time. If you need a dynamically sized array, std::vector is a better choice.
*/

/*
C++20 span 
- enables programs to view contiguous elements of a container (built-in array, std::array, std::vector)
- “sees” the container’s elements but does NOT have its own copy of those elements

.. guidelines:
- pass built-in arrays to functions as spans, which contain both a pointer to the array’s 1st element and the array’s size
- pass a span by value as it’s just as efficient as passing a pointer and size separately
*/

#include <array>
#include <format>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>
#include <print> // C++23 header for std::print and std::println

// items parameter is treated as a const int* so we also need the size to
// know how to iterate over items with counter-controlled iteration
void displayArray(const int items[], size_t size) { //int items[] decays to int* automatically
   for (size_t i{0}; i < size; ++i) {
      std::cout << std::format("{} ", items[i]);
   }
}

// span parameter contains both the location of the first item
// and the number of elements, so we can iterate using range-based for
void displaySpan(std::span<const int> items) {
   for (const auto& item : items) { // spans are iterable
      std::cout << std::format("{} ", item);
   }
}

// spans can be used to modify elements in the original data structure
void times2(std::span<int> items) {
   for (int& item : items) { //note that const is not here anymore as compared to function above
      item *= 2;
   }
}

int main() {
   int values1[]{1, 2, 3, 4, 5}; //built-in C-style array
   std::array values2{6, 7, 8, 9, 10};
   std::vector values3{11, 12, 13, 14, 15};

   //int* danglingPtr; // uninitialized “dangling pointer”
   int* startPtr{nullptr}; // pointer to nothing // NOT used below
   //do NOT use these null pointers from before C++11: 0, NULL 
   /*
•	Type Safety:
nullptr has its own type (std::nullptr_t) and is not implicitly convertible to integer types. This prevents accidental type mismatches and errors, which can occur when using NULL (often defined as 0).
   */

   // must specify size because the compiler treats displayArray's items 
   // parameter as a pointer to the first element of the argument
   std::cout << "values1 via displayArray {built-in C-style array passed thru const int items[], size_t size}: ";
   displayArray(values1, 5);

   // compiler knows values1's size and automatically creates a span
   // representing &values1[0] and the array's length
   // note that &values1[0] above is the address/pointer to the 1st element of the values1 array
   std::cout << "\nvalues1 via displaySpan [built-in C-style array passed thru std::span<const int> items]: ";
   displaySpan(values1);

   // compiler also can create spans from std::arrays and std::vectors
   std::cout << "\nvalues2 via displaySpan [std::array passed thru std::span<const int> items]: ";
   displaySpan(values2);
   std::cout << "\nvalues3 via displaySpan [std::vector passed thru std::span<const int> items]: ";
   displaySpan(values3);

   // changing a span's contents modifies the ORIGINAL data
   times2(values1);
   std::cout << "\n\nvalues1 after times2 modifies its span argument: ";
   displaySpan(values1);

   // spans have various array-and-vector-like capabilities
   std::span mySpan{values1}; // span<int>
   std::cout << "\n\nFirst element of std::span mySpan{values1} is [mySpan.front()]: " << mySpan.front()
      << "\nmySpan's last element [mySpan.back()]: " << mySpan.back();

   // spans can be used with standard library algorithms
   std::cout << "\n\nSum of mySpan's elements [using accumulate and iterators]: "
      << std::accumulate(std::begin(mySpan), std::end(mySpan), 0);

   // spans can be used to create subviews of a container
   std::cout << "\n\nFirst three elements of mySpan [mySpan.first(3)]: ";
   displaySpan(mySpan.first(3));
   std::cout << "\nLast three elements of mySpan [mySpan.last(3)]: ";
   displaySpan(mySpan.last(3));
   std::cout << "\nMiddle three elements of mySpan [mySpan.subspan(1, 3)]: ";
   displaySpan(mySpan.subspan(1, 3)); //start from position 1, for 3 elements

   // changing a subview's contents modifies the original data
   times2(mySpan.subspan(1, 3));
   std::cout << "\n\nvalues1 after modifying a subset of the elements via span [times2(mySpan.subspan(1, 3))]: ";
   displaySpan(values1);

   // access a span element via []
   std::cout << "\n\nThe element at index 2 is (span element accessed via mySpan[2]): " << mySpan[2];

   std::println("\nTHE END");
   return 0;
}