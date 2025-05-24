/*
From https://en.cppreference.com/w/cpp/language/integer_literal.html :
z or Z    - the signed version of std::size_t (since C++23)
both z/Z and u/U    - std::size_t (since C++23)

note: "uz" used below

From https://en.cppreference.com/w/cpp/types/size_t.html :
std::size_t can store the maximum size of a theoretically possible object of any type (including array). On many platforms (an exception is systems with segmented addressing) std::size_t can safely store the value of any non-member pointer, in which case it is synonymous with std::uintptr_t.
std::size_t is commonly used for array indexing and loop counting. Programs that use other types, such as unsigned int, for array indexing may fail on, e.g. 64-bit systems when the index exceeds UINT_MAX or if it relies on 32-bit modular arithmetic.

*/
#include <iostream>
#include <vector>
#include <numeric> //for using std::iota
#include <climits> // Required for CHAR_BIT
#include <locale> // to print large numbers with commas (for U.S. locale, in this case)

#include <iomanip> // for setw and put_money

using namespace std;

int main() {
    int times_printed{ 0 };
    bool printed_i_width{ false };
    cout.imbue(std::locale("")); // Use the system's default locale
	locale us_locale = locale("en_US.UTF-8"); // Store the default locale for later use in format

    long longNum;
    int longBits{ sizeof(longNum) * CHAR_BIT };
    cout << "Bit width of long: " << longBits << endl; // 32 in the env tried
    long long longlongNum;
    int longlongBits = sizeof(longlongNum) * CHAR_BIT;
    cout << "Bit width of long long: " << longlongBits << endl; // 64 in the env tried

    // Create a vector of integers with a size of 10 billion and 1.
    vector<long long> myVector(10'000'000'001); // single quote marks are ignored when determining value
    cout << "\nCreated myVector of size: " << myVector.size() << endl;

    // Initialize the vector with values from 1 to 10,000,000,001 using std::iota.
    iota(myVector.begin(), myVector.end(), 1ULL);
    //iota(myVector.begin(), myVector.end(), 1); //this does NOT work as expected

    // Example: Print the first 10 elements.
	cout << "First 10 elements: ";
    for (int i = 0; i < 10; ++i) {
        cout << myVector[i] << " ";
    }
    cout << "\n";

    // Example: Print the last 10 elements.
	cout << "Last 10 elements: ";
    for (long long i = myVector.size() - 10; i < myVector.size(); ++i) { // will print the intended last 10 elements
    //for (long i = myVector.size() - 10; i < myVector.size(); i++) { // printed 20 elements
    //for (int i = myVector.size() - 10; i < myVector.size(); i++) { // printed 20 elements
        cout << myVector[i] << " ";
        times_printed++;
        if (times_printed == 20) {
            break;
        }
    }
    cout << endl << endl;

    // Infinite loop if myVector.size > max unsigned int (= 2^32 – 1, which is: 4 billion – 1)
    times_printed = 0;
    for (auto i = 0u; i < myVector.size(); ++i)
    {
        if (printed_i_width == false) {
            size_t iBits = sizeof(i) * CHAR_BIT;
            cout << "Bit width of index i (for loop with: auto i = 0; i < myVector.size(); ++i): " << iBits << "\n"; // should be 32
            cout << ".. hence running into infinite loop which now got extra code to break after printing 20 times\n";
            printed_i_width = true;
        }
        if ((i + 1) % 1000000000 == 0) {
			cout << "Value at index " << i << ": " << myVector.at(i) << endl; // using .at here instead, bound-checking at compile time but slower than [] operator
            times_printed++;
        }
        if (times_printed == 20) {
            break;
        }
    }

    times_printed = 0;
    printed_i_width = false;
    for (auto i = 0u; i < myVector.size(); ++i)
    { 
        if (printed_i_width == false) {
            size_t iBits = sizeof(i) * CHAR_BIT;
            cout << "\nBit width of index i (for loop with: auto i = 0u; i < myVector.size(); ++i): " << iBits << "\n"; // should be 32
            cout << ".. hence running into infinite loop which now got extra code to break after printing 20 times\n";
            printed_i_width = true;
        }
        if ((i + 1) % 1000000000 == 0) {
            cout << "Value at index " << i << ": " << myVector[i] << endl;
            times_printed++;
        }
        if (times_printed == 20) {
            break;
        }
    } 

    // Fixed because of uz literal 
    cout << "\nPrinting the billions correctly (because C++23 uz used: auto i = 0uz):\n";
    printed_i_width = false;
    for (auto i = 0uz; i < myVector.size(); ++i)
    { 
        if (printed_i_width == false) {
            size_t iBits = sizeof(i) * CHAR_BIT;
            cout << ".. Bit width of index i here: " << iBits << "\n\n"; // should be 64
            printed_i_width = true;
        }
        if ((i + 1) % 1'000'000'000 == 0) {
            cout << "Value at index " << i << ": " << myVector[i] << endl;
        }
    } 

    cout << "\nPrinting the billions correctly (because ull used: auto i = 0ull) while using C++17 format to right-align:\n";
    printed_i_width = false;
    for (auto i = 0ull; i < myVector.size(); ++i)
    {
        if (printed_i_width == false) {
            size_t iBits = sizeof(i) * CHAR_BIT;
            cout << ".. Bit width of index i here: " << iBits << "\n\n"; // should be 64
            printed_i_width = true;
        }
        if ((i + 1) % 1'000'000'000 == 0) {
            //cout << "Value at index " << i << ": " << myVector[i] << endl;
			cout << format(us_locale, "Value at index {:>15L}: {:>20L}\n", i, myVector[i]); // note use of "L" for locale-specific form presentation type
        }
    }

    cout << "\nPrinting the billions correctly (because size_t [since C++11] used: size_t i = 0):\n";
    printed_i_width = false;
    for (size_t i = 0; i < myVector.size(); ++i)
    {
        if (printed_i_width == false) {
            size_t iBits = sizeof(i) * CHAR_BIT;
            cout << ".. Bit width of index i here: " << iBits << "\n\n"; // should be 64
            printed_i_width = true;
        }
        if ((i + 1) % 1'000'000'000 == 0) {
            cout << "Value at index " << i << ": " << myVector[i] << endl;
        }
    }

    cout << "\nTHE END";
  
    return 0;
}
