//note that these are NOT covered below (at least not explicitly): 
//.joinable()/.join() vs .detach()/system(“Pause”)
//std::ref/cref (for passing arguments to std::thread functions by reference)
//HANDLE
//SetThreadDescription()
//get_id()
//std::launch::deferred (the launch policy std::launch::async explicitly used below to create a new thread)
//std::promise which is useful for sharing data between threads without having to manually perform thread synchronization
//.. get_future(), set_value()
//exception handling - like propagating an exception from one thread to another by creating an exception pointer using std::make_exception_ptr and setting it in the Promise using set_exception

//std:;async - https://en.cppreference.com/w/cpp/thread/async.html
// The function template std::async runs the function f asynchronously (potentially in a separate thread which might be a part of a thread pool) and returns a std::future that will eventually hold the result of that function call.

//std::future - https://en.cppreference.com/w/cpp/thread/future.html
/*
The class template std::future provides a mechanism to access the result of asynchronous operations:
	An asynchronous operation (created via std::async, std::packaged_task, or std::promise) can provide a std::future object to the creator of that asynchronous operation.
	The creator of the asynchronous operation can then use a variety of methods to query, wait for, or extract a value from the std::future. These methods may block if the asynchronous operation has not yet provided a value.
	When the asynchronous operation is ready to send a result to the creator, it can do so by modifying shared state (e.g. std::promise::set_value) that is linked to the creator's std::future.
Note that std::future references shared state that is not shared with any other asynchronous return objects (as opposed to std::shared_future).
*/

// std::future<T>::get - https://en.cppreference.com/w/cpp/thread/future/get
// The get member function waits (by calling wait()) until the shared state is ready, then retrieves the value stored in the shared state (if any). Right after calling this function, valid() is false.
// Notes - The C++ standard recommends the implementations to detect the case when valid() is false before the call and throw a std::future_error with an error condition of std::future_errc::no_state.
// .. NOT done below for simplicity

/*
std::list and std::thread together can lead to this issue and how to address it :
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

//std::list<int> g_Data; //g for global //from modernCPP_thread_unsafe.cpp, keeping here as a reference
std::list<int> g_Data_mutex; //g for global
const int SIZE = 100000;
std::mutex g_Mutex; //g for global

/* from modernCPP_thread_unsafe.cpp, keeping here as a reference
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
*/

void Download_mutex() {
	for (int i = 0; i < SIZE; ++i) {
		//Use std::lock_guard to lock a mutex (RAII)
		//.. instead of using *.lock()/*.unlock(), which can be unsafe, like when the thread function returns early unexpectedly before unlocking
		std::lock_guard<std::mutex> mtx(g_Mutex); //mtx is a local object that is automatically destroyed on this function’s exit, and such would automatically unlocks the mutex
//i.e. lock_guard automatically locks the mutex in its constructor and unlocks it in its destructor
		g_Data_mutex.push_back(i);
	}
}
void Download2_mutex() {
	for (int i = 0; i < SIZE; ++i) {
		std::lock_guard<std::mutex> mtx(g_Mutex);
		g_Data_mutex.push_back(i);
	}
}

int asyncLongTask(int count) {
	using namespace std::chrono_literals;
	int sum{0};
	std::cout << "\nJust went into asyncLongTask with input count of " << count << std::flush << std::endl;
	for (int i = 0; i < count; ++i) {
		sum += i;
		std::cout << ".. from asyncLongTask with input count of " << count << " and i (up to count-1) is " << i << std::flush << std::endl;
		std::this_thread::sleep_for(500ms);//or longer, like: 1s or std::chrono::seconds(1) 
	}
	return sum; //sum is the shared state here, accessed through the future object below
}

int main() {
	using namespace std::chrono_literals;
	int count{10}; //count will be passed below to the function asyncLongTask above
	auto cores {std::thread::hardware_concurrency()};
	std::cout << "Number of cores:" << cores << std::endl;

	/* from modernCPP_thread_unsafe.cpp, keeping here as a reference
	std::thread thDownloader(Download);
	std::thread thDownloader2(Download2);
	thDownloader.join();
	thDownloader2.join();
	std::cout << "2 threads, each writing " << SIZE << " (increase SIZE if needed) times to g_Data which only resulted in a size of: " << g_Data.size() << std::endl;
	std::cout << ".. not getting expected result of " << SIZE * 2 << " due to race condition\n\n";
	*/

	std::thread thDownloader_mutex(Download_mutex);
	std::thread thDownloader2_mutex(Download2_mutex);
	thDownloader_mutex.join();
	thDownloader2_mutex.join();
	std::cout << "\nNow getting [compare to race condition encountered in modernCPP_thread_unsafe.cpp] this correctly: ";
	std::cout << g_Data_mutex.size() << std::endl;
	std::cout << ".. which is same as expected result of " << SIZE * 2 << " (by using: mutex)\n\n";
	std::cout << ".. However, locking/unlocking this way is not efficient, as a thread got to wait while waiting to lock successfully\n";
	std::cout << ".. so, each thread could have its own list here and populate it, then combine the lists when all threads are done\n\n";

	std::cout << "Now, switching from low-level concurrency to high-level concurrency and use async/future\n";
	std::cout << "Async function will execute a function, or a callable, in a separate thread automatically without directly managing threads.\n\n";
	
	std::future<int> result = std::async(std::launch::async, asyncLongTask, count);
//"if" the std::launch::deferred launch policy is used instead, then this async task will be launched when .get() executes below

	std::cout << "main loop in main() thread about to start execution .. note that messages printed from 2 threads can be mixed together ..\n";
	for (int i = 0; i < count; ++i) {
		std::cout << ".. from main() .. iteration " << i+1 << " out of " << count << std::flush << std::endl;
		std::this_thread::sleep_for(300ms);//or longer, like: 1s or std::chrono::seconds(1) 
	}
	std::cout << "\nmain loop of main() thread just finished execution...\n\n";

	if (result.valid()) {
		auto sum = result.get(); // .get() will make the main thread waits for the shared state, sum, from the async thread
		std::cout << "Getting final result from the async thread (which is the sum of 0, 1, up to " << count-1 << "): " << sum << std::endl;
	}

	std::cout << "\nFinal experiment: executing multiple async tasks .. print messages from various async tasks can be mixed together ..\n";
	std::vector<std::future<int>> asynctasks;
	int main_sum{0};
	for (int i = 0; i < count+1; i += 5) {
		asynctasks.push_back( std::async(std::launch::async, asyncLongTask, i) ); //0, 0+1+2+3+4=10, 0+1+2..+9=45
		for (int j = 0; j < i; ++j) {
			main_sum += j;
		}
	}
	auto total{0};
	for (auto& task : asynctasks) {
		total += task.get();
	}
	std::print("\nSumming up the results from all async tasks .. expecting {} and got ", main_sum);
	std::cout << total << std::endl; //0 + 10 (0+1+2+3+4) + 45 (0+1+2 .. + 9) = 55

	std::println("\nTHE END");
	return 0;
}