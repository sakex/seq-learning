//
// Created by alexandre on 26.03.20.
//

#ifndef SEQ_LEARNING_THREAD_N_M_HPP
#define SEQ_LEARNING_THREAD_N_M_HPP

#include <thread>
#include <functional>
#include <mutex>

namespace threading {
    template<typename CallableIi, typename CallableIj>
    void thread_n_m(uint16_t const n, CallableIi &cb_ii, CallableIj &cb_ij,
                    unsigned max_threads = std::thread::hardware_concurrency()) {
        max_threads--;
        std::mutex locker;
        std::vector<std::thread> threads;
        int *_i = new int(0);
        int *_j = new int(0);
        // Making sure we are on the heap
        auto lambda = [n, _i, _j, &cb_ii, &cb_ij, &locker]() -> void {
            int &i = *_i;
            int &j = *_j;
            while (true) {
                locker.lock();
                int temp_i = i;
                int temp_j = j;
                if (temp_j == n - 1) {
                    j = ++i;
                } else {
                    j++;
                }
                locker.unlock();
                if (temp_i < n) {
                    if (temp_i == temp_j) cb_ii(temp_i);
                    else cb_ij(temp_i, temp_j);
                } else
                    break;
            }
        };
        for (size_t it = 0; it < max_threads; ++it)
            threads.push_back(std::thread(lambda));
        lambda(); // run on main thread
        for (std::thread &thread : threads) {
            thread.join();
        }
        delete _i;
        delete _j;
    }
}

#endif //SEQ_LEARNING_THREAD_N_M_HPP
