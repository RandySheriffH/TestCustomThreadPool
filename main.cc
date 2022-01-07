#include "onnxruntime_cxx_api.h"
#include "tbb/parallel_for.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <deque>
#include <functional>

struct ThreadPoolFoo: public Ort::OrtThreadPool {

    ThreadPoolFoo(int num_threads) {
        tasks_.resize(num_threads);
        states_.reset(new AtomicState[num_threads]);
        for (int i = 0; i < num_threads; ++i) {
            threads_.push_back(std::thread([this, i](){
                while (!quit_) {
                    State state_ready{State::ready};
                    if (states_[i].compare_exchange_weak(state_ready, State::running, std::memory_order_acquire)) {
                        if (tasks_[i]) tasks_[i]();
                        states_[i].store(State::done, std::memory_order_release);
                    } else {
                        std::this_thread::yield();
                    }
                }
            }));
        }
    }
    
    ~ThreadPoolFoo() {
        quit_ = true;
        for (auto& t: threads_) t.join();
    }
    
    void ParallelFor(std::ptrdiff_t iterations, const std::function<void(std::ptrdiff_t from, std::ptrdiff_t to)>* func) override {
        std::cout << "foo ParallelFor called!" << std::endl;
        std::atomic<ptrdiff_t> iterator{0};
        ptrdiff_t block_size{2};
        std::vector<int> engaged;
        Task task = [&] () {
            ptrdiff_t from = 0;
            while ((from = iterator.fetch_add(block_size, std::memory_order_relaxed)) < iterations) {
                (*func)(from, std::min(from + block_size, iterations));
            }
        };
        for (int i = 0; i < NumThreads(); ++i) {
            State state_available{State::available};
            if (states_[i].compare_exchange_weak(state_available, State::loading, std::memory_order_relaxed)) {
                engaged.push_back(i);
                tasks_[i] = task;
                states_[i].store(State::ready, std::memory_order_release);
            }
        }
        task();
        for (auto i: engaged) {
            State state_ready{State::ready};
            if (!states_[i].compare_exchange_weak(state_ready, State::available, std::memory_order_relaxed)) {
                State state_done{State::done};
                while (!states_[i].compare_exchange_weak(state_done, State::available, std::memory_order_acquire)) {
                    state_done = State::done;
                    std::this_thread::yield();
                }
            }
        }
    }

    int NumThreads() const override {
        std::cout << "foo NumThreads called!" << std::endl;
        return threads_.size();
    }
    
    std::vector<std::thread> threads_;
    using Task = std::function<void()>;
    std::vector<Task> tasks_;
    enum class State {
        available = 0,
        loading,
        ready,
        running,
        done
    };
    using AtomicState = std::atomic<State>;
    std::unique_ptr<AtomicState[]> states_;
    bool quit_ = false;
}; //ThreadPoolFoo

struct ThreadPoolTbb : public Ort::OrtThreadPool {
    ThreadPoolTbb(int num_threads) : num_threads_(num_threads) {};
    ~ThreadPoolTbb() = default;
    using Fn = std::function<void(std::ptrdiff_t from, std::ptrdiff_t to)>;

    struct ParallelClass {
        ParallelClass(const Fn* fn) : fn_(fn) {}
        void operator()(tbb::blocked_range<std::ptrdiff_t>& range)const {
            std::cout << "tbb ParallelFor called!" << std::endl;
            (*fn_)(range.begin(), range.end());
        }
        const Fn* fn_{};
    };

    void ParallelFor(std::ptrdiff_t iterations, const Fn* fn) override {
        tbb::parallel_for(tbb::blocked_range<std::ptrdiff_t>(0, iterations), ParallelClass(fn));
    }

    int NumThreads() const override {
        return num_threads_;
    }

    int num_threads_ = 0;
};

void TestTP() {

    constexpr int buf_size = 1024;
    int value = 3;
    int A[buf_size];
    int B[buf_size];

    for (int i = 0; i < buf_size; ++i) {
        A[i] = value;
        B[i] = 0;
    }

    std::function<void(ptrdiff_t, ptrdiff_t)> func = [&](ptrdiff_t from, ptrdiff_t to) {
        for (int i = from; i < to; ++i) B[i] = A[i];
    };

    ThreadPoolFoo tp_foo(4);

    tp_foo.ParallelFor(buf_size, &func);

    try {
        for (int i = 0; i < buf_size; ++i) {
            if (B[i] != 3) {
                throw std::exception();
            }
        }
    }
    catch (...) {
        std::cout << "mismatch!" << std::endl;
    }

    std::cout << B[0] << ", " << B[1] << ", " << B[2] << "..." << std::endl;
    std::cout << "done" << std::endl;
}

template<typename ThreadPoolType>
void TestAdd() {
    ThreadPoolType tp_instance(4);
    std::cout << "test custom thread pool." << std::endl;
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
    Ort::SessionOptions session_options;
    session_options.SetSessionThreadPool(&tp_instance);
    Ort::Session session{ env, L"D:\\issue\\CustomThreadPool\\model\\model.onnx", session_options };
    //Ort::Session session{ env, L"D:\\issue\\CustomThreadPool\\model\\model.onnx", Ort::SessionOptions{nullptr} };
    //session.SetThreadPool(&tp_foo);
    std::cout << "loaded" << std::endl;
    const char* input_names[] = {"X", "Y"};
    Ort::AllocatorWithDefaultOptions allocator_info;
    constexpr int dim = 1024;
    int32_t ints[dim];
    for (int i = 0; i < dim; ++i) ints[i] = 1;
    int64_t shape[] = {dim};
    const char* output_names[] = {"Z"};
    Ort::Value input_tensors[] = {Ort::Value::CreateTensor<int32_t>(allocator_info.GetInfo(), ints, dim, shape, 1), 
                                  Ort::Value::CreateTensor<int32_t>(allocator_info.GetInfo(), ints, dim, shape, 1)};
    Ort::Value output_tensors[] = {Ort::Value::CreateTensor<int32_t>(allocator_info.GetInfo(), ints, dim, shape, 1)};
    session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors, 2, output_names, output_tensors, 1);
    const int32_t* output_data = output_tensors[0].GetTensorData<int32_t>();
    for (int i = 0; i < 3; ++i) {
        std::cout << output_data[i] << ", ";
    }
    std::cout << "..." << std::endl;
    std::cout << "done" << std::endl;
}

void TestPGAN() {
    ThreadPoolFoo tp_foo(4);
    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "test" };
    Ort::SessionOptions session_options;
    session_options.SetSessionThreadPool(&tp_foo);
    Ort::Session session{ env, L"D:\\issue\\PerfInvestigation\\pgan\\PGAN_NetG_model.onnx", session_options };
    //Ort::Session session{ env, L"D:\\issue\\PerfInvestigation\\pgan\\PGAN_NetG_model.onnx", Ort::SessionOptions{nullptr} };
    //session.SetThreadPool(&tp_foo);
    std::cout << "loaded" << std::endl;
    const char* input_names[] = { "0" };
    Ort::AllocatorWithDefaultOptions allocator_info;
    constexpr int input_dim = 2*2*16*16;
    float* input_floats = new float[input_dim];
    int64_t input_shape[] = { 2,2,16,16 };
    Ort::Value input_tensors[] = { Ort::Value::CreateTensor<float>(allocator_info.GetInfo(), input_floats, input_dim, input_shape, 4) };

    const char* output_names[] = { "566" };
    constexpr int output_dim = 2 * 3 * 256 * 256;
    float* output_floats = new float[output_dim];
    int64_t output_shape[] = {2, 3, 256, 256};
    Ort::Value output_tensors[] = { Ort::Value::CreateTensor<float>(allocator_info.GetInfo(), output_floats, output_dim, output_shape, 4) };

    session.Run(Ort::RunOptions{ nullptr }, input_names, input_tensors, 1, output_names, output_tensors, 1);

    const float* output_data = output_tensors[0].GetTensorData<float>();
    for (int i = 0; i < 3; ++i) {
        std::cout << output_data[i] << ", ";
    }
    std::cout << "..." << std::endl;
    std::cout << "done" << std::endl;
    delete[] input_floats;
    delete[] output_floats;
}

int main() {
    //TestAdd<ThreadPoolFoo>();
    TestAdd<ThreadPoolTbb>();
    //TestPGAN();
}