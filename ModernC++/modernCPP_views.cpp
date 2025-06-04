//iota - generates a sequence of elements by repeatedly incrementing an initial value

// std::views in C++20 offer a powerful way to work with ranges of data
//Lazy Evaluation: Views perform transformations and filtering on-demand, only when an element is accessed. 
//This avoids unnecessary computations and memory allocation, especially beneficial when dealing with large datasets or complex operations.

// views::filter – selects only the range elements that match a condition

// | operator to form a pipeline of operations

#include <array>
#include <format>
#include <iostream>
#include <numeric>
#include <ranges>
#include <print> // C++23 header for std::print and std::println

int main() {

   // lambda to display results of range operations
   auto showValues{
      [](auto& values, const std::string& message) {
         std::cout << std::format("{}: ", message);

         for (const auto& value : values) {
            std::cout << std::format("{} ", value);
         }

         std::cout << '\n';
      }
   };

   auto values1{std::views::iota(1, 11)}; // generate integers 1-10
   showValues(values1, "Generate integers 1-10 [auto values1{std::views::iota(1, 11)}]");

   // filter each value in values1, keeping only the even integers
   auto values2{values1 | std::views::filter([](const auto& x) {return x % 2 == 0;})};
   showValues(values2, "Filtering even integers [using | op and std::views::filter with lambda]");

   // map each value in values2 to its square
   auto values3{values2 | std::views::transform([](const auto& x) {return x * x;})};
   showValues(values3, "Mapping even integers to squares [using | op and std::views::transform with lambda]");

   // combine filter and transform to get squares of the even integers
   auto values4{
      values1 | std::views::filter([](const auto& x) {return x % 2 == 0;})
              | std::views::transform([](const auto& x) {return x * x; })};
   showValues(values4, "Squares of even integers [using | op and std::views::filter/transform with lambda]");

   // total the squares of the even integers 
   std::cout << std::format("Sum squares of even integers 2-10 [using std::accumulate with iterators]: {}\n",
      std::accumulate(std::begin(values4), std::end(values4), 0));

   // process a container's elements
   constexpr std::array numbers{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
   auto values5{
      numbers | std::views::filter([](const auto& x) {return x % 2 == 0;})
              | std::views::transform([](const auto& x) {return x * x;})};
   showValues(values5, "Squares of even integers in array numbers [using | op and std::views::filter/transform with lambda on: constexpr std::array]");

   std::println("\nTHE END");
   return 0;
}
