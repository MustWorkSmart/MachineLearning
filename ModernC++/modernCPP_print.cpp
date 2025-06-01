// print and println from C++23
// https://en.cppreference.com/w/cpp/io/println.html
// compile-time format checking, performance

// note: 
// Python's f-strings offer direct embedding of variables and expressions within the string, 
// while C++23's std::print requires separate arguments.

#include <iostream>
#include <print>

int main() {
    int mynum{42};

    //before c++23:
    std::cout << "..before C++23.." << std::endl;
    std::cout << "Hello!\n";
    
    //c++23:
    std::print("\n..with C++23 print and println:");
    std::print("\nHello, {}!\n", "world");
    std::println("Hello from C++23!"); // no need to use stream ops (<<) or manually insert ‘\n’ or std::endl
    
    // std::println uses compile-time format checking, similar to std::format, which catches errors early
    std::println("Value: {}", mynum); // OK
    std::println("Value: {}", "text"); // OK
    //std::println("Value: {} {}", 42); // Compile-time error

    std::cout << "\nTHE END";
  
    return 0;
}
