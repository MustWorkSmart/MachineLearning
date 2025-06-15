/*
std::list and std::thread together can lead to runtime library assertion failure .. and how to address it :

Thread Safety and std::list :
	std::list(and other standard library containers like std::vector and std::map) are not thread - safe by default.
	If multiple threads access(read or write) the same std::list without proper synchronization, you can encounter data races and undefined behavior, leading to memory corruption.
	Even read operations are not always thread - safe if another thread is modifying the list concurrently.
	Heap Corruption and _CrtIsValidHeapPointer :
	The _CrtIsValidHeapPointer assertion specifically checks if a pointer points to memory that was allocated on the heap by the C runtime library.
	When multiple threads access a non - thread - safe container like std::list without synchronization, it can lead to situations where :
A thread tries to access an element that has been freed by another thread.
A thread tries to free an element that has already been freed by another thread(double deletion).
The internal structure of the std::list becomes corrupted due to concurrent modifications, leading to invalid pointers.
*/

#include <iostream>
#include <list>
#include <thread>
#include <future> //for std::async which is used below, std::async returns an object of type std::future
#include <string>
#include <mutex> // mutex is used to synchronize access to shared data between multiple threads, preventing race conditions where data may be accessed simultaneously
#include <print> // C++23 header for std::print and std::println
#include <chrono>

std::list<int> g_Data; //g for global
const int SIZE = 100000;

void Download() {
	for (int i = 0; i < SIZE; ++i) {
		g_Data.push_back(i);
	}
}
void Download2() {
	for (int i = 0; i < SIZE; ++i) {
		g_Data.push_back(i);
	}
}

int main() {
	using namespace std::chrono_literals;
	int count{10}; //count will be passed below to the function asyncLongTask above
	auto cores {std::thread::hardware_concurrency()};
	std::cout << "Number of cores:" << cores << std::endl;

	std::thread thDownloader(Download);
	std::thread thDownloader2(Download2);
	thDownloader.join();
	thDownloader2.join();
	std::cout << "\n2 threads, each writing " << SIZE << " (increase SIZE if needed) times to g_Data which only resulted in a size of: " << g_Data.size() << std::endl;
	std::cout << ".. not getting expected result of " << SIZE * 2 << " due to race condition\n\n";

	std::println("\nTHE END .. although runtime library assertion failure would likely be there as well ..");
	return 0;
}