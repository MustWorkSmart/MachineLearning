//using accumulate to compute the sum/product of elements in an array, constexpr and iterators also used

//The std::accumulate function in C++ is a powerful tool for performing cumulative operations on a range of elements. 
//It is part of the <numeric> header and provides a generic way to compute sums, products, or other aggregated values.

/*
begin() and end() are functions (or methods, depending on context) used to obtain iterators that define the range of a container or a sequence. 
They are fundamental for working with various data structures and algorithms in the C++ Standard Template Library (STL).

begin():
Returns an iterator pointing to the first element of the container or sequence. 
If the container is empty, begin() returns the same iterator as end(). 

end():
Returns an iterator pointing to the theoretical element one position past the last element of the container. 
It does not point to a valid element and is used as a sentinel value to indicate the end of the sequence. 

e.g.
    const int x = 10; // x is const and initialized with a constant value at compile-time
    int y = 20;
    const int z = y; // z is const but initialized with a runtime variable
*/

/*
constexpr vs const

const
Meaning: const indicates that a variable's value will not be changed after initialization.
Initialization: const variables can be initialized at runtime.

constexpr
Meaning: constexpr indicates that a variable's value must be known at compile time and can be used in constant expressions.
Initialization: constexpr variables must be initialized with constant expressions at compile time.
*/

#include <array>
#include <format>
#include <iostream>
#include <numeric>
#include <ranges>
#include <print> // C++23 header for std::print and std::println

int multiply(int x, int y) {
   return x * y;
}

int main() {
   constexpr std::array integers{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
   std::println("Total of array elements [using std::accumulate]: {}", //using println instead of cout below
   //std::cout << std::format("Total of array elements [using std::accumulate]: {}\n",
      std::accumulate(std::begin(integers), std::end(integers), 0)); //expecting: 55
   // 0 above is the initial value to be added on

   constexpr std::array ints12345{1, 2, 3, 4, 5};

   std::cout << std::format("Product of integers [using accumulate with custom function]: {}\n", 
      std::accumulate(std::begin(ints12345), std::end(ints12345), 1, multiply)); //expecting: 120
   // 1 above is the initial value to be multiplied with

   std::cout << std::format("Product of integers [using std::accumulate and lambda]: {}\n",
      std::accumulate(std::begin(integers), std::end(integers), 1,
         [](const auto& x, const auto& y) {return x * y;}));
   // [] above is indicating the start of the inline lambda expression

    std::println("\nTHE END");
    return 0;
}
