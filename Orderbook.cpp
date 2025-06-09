// Ultra Low Latency Orderbook - Final Working Version
#include <atomic>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <new>
#include <iostream>
#include <functional>
#include <chrono>
#include <stdlib.h>
#include <unordered_map>  // Using std::unordered_map instead of tsl::robin_map

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
    #define OS_WINDOWS 1
#elif defined(__linux__)
    #define OS_LINUX 1
#elif defined(__APPLE__)
    #define OS_MAC 1
#else
    #define OS_UNKNOWN 1
#endif

// Prefetch macros
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    #include <x86intrin.h>
    #define PREFETCH(addr) __builtin_prefetch(addr)
    #define PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1)
#elif defined(__GNUC__) || defined(__clang__)
    #define PREFETCH(addr) __builtin_prefetch(addr)
    #define PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1)
#elif defined(_MSC_VER)
    #include <intrin.h>
    #define PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
    #define PREFETCH_WRITE(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T1)
#else
    #define PREFETCH(addr) ((void)0)
    #define PREFETCH_WRITE(addr) ((void)0)
#endif

// Cache line size
#ifndef hardware_destructive_interference_size
    #if defined(_MSC_VER)
        #define hardware_destructive_interference_size 64
    #else
        #define hardware_destructive_interference_size 64
    #endif
#endif

namespace Orderbook {

constexpr size_t CACHE_LINE_SIZE = hardware_destructive_interference_size;
constexpr size_t MAX_LEVELS = 1024;
constexpr size_t INITIAL_ORDER_CAPACITY = 4096;

enum class OrderType : uint8_t { LIMIT = 0, MARKET = 1, IOC = 2 };
enum class Side : uint8_t { BUY = 0, SELL = 1 };
enum class OrderStatus : uint8_t { OPEN = 0, PARTIALLY_FILLED = 1, FILLED = 2, CANCELLED = 3, REJECTED = 4 };

using Price = int64_t;
using Quantity = int64_t;
using OrderId = uint64_t;
using Timestamp = uint64_t;

// Aligned allocator
template<typename T, size_t Alignment = CACHE_LINE_SIZE>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using size_type = std::size_t;

    template<typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    pointer allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(value_type)) {
            throw std::bad_alloc();
        }

        void* ptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(n * sizeof(value_type), Alignment);
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(value_type))) {
            throw std::bad_alloc();
        }
#endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        new((void*)p) U(std::forward<Args>(args)...);
    }

    template<typename U>
    void destroy(U* p) {
        p->~U();
    }
};

template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

// Order structure
struct alignas(CACHE_LINE_SIZE) Order {
    OrderId id;
    Price price;
    Quantity quantity;
    Quantity filled_quantity;
    Side side;
    OrderType type;
    std::atomic<OrderStatus> status;
    Timestamp timestamp;
    
    Order(OrderId i, Price p, Quantity q, Side s, OrderType t, Timestamp ts)
        : id(i), price(p), quantity(q), filled_quantity(0), side(s), type(t), 
          status(OrderStatus::OPEN), timestamp(ts) {}
    
    Order(const Order&) = delete;
    Order& operator=(const Order&) = delete;
    Order(Order&&) = default;
    Order& operator=(Order&&) = default;
    
    bool operator<(const Order& other) const noexcept {
        if (side == Side::BUY) {
            return price > other.price || (price == other.price && timestamp < other.timestamp);
        }
        return price < other.price || (price == other.price && timestamp < other.timestamp);
    }
};

// Level structure
struct alignas(CACHE_LINE_SIZE) Level {
    Price price;
    AlignedVector<Order*> orders;
    std::atomic<Quantity> total_quantity;
    
    Level() : price(0), total_quantity(0) {}
    
    Level(Level&& other) noexcept 
        : price(other.price),
          orders(std::move(other.orders)),
          total_quantity(other.total_quantity.load(std::memory_order_relaxed)) {}
    
    Level& operator=(Level&& other) noexcept {
        if (this != &other) {
            price = other.price;
            orders = std::move(other.orders);
            total_quantity.store(other.total_quantity.load(std::memory_order_relaxed), 
                               std::memory_order_relaxed);
        }
        return *this;
    }
    
    void prefetch() const noexcept {
        PREFETCH(this);
        if (!orders.empty()) PREFETCH(orders.data());
    }
};

class Orderbook {
private:
    alignas(CACHE_LINE_SIZE) std::atomic<OrderId> sequence_id_{0};
    alignas(CACHE_LINE_SIZE) std::unordered_map<OrderId, std::unique_ptr<Order>> orders_map_;
    alignas(CACHE_LINE_SIZE) AlignedVector<Level> bids_;
    alignas(CACHE_LINE_SIZE) AlignedVector<Level> asks_;

    OrderId generate_id() noexcept {
        return sequence_id_.fetch_add(1, std::memory_order_relaxed);
    }
    
    Timestamp get_timestamp() const noexcept {
#if defined(__x86_64__) || defined(__i386__)
        return __builtin_ia32_rdtsc();
#elif defined(_MSC_VER) && defined(_M_IX86)
        return __rdtsc();
#else
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif
    }
    
    Level* find_or_create_level(Price price, Side side) {
        auto& levels = side == Side::BUY ? bids_ : asks_;
        PREFETCH(levels.data());
        
        auto it = std::lower_bound(levels.begin(), levels.end(), price,
            [side](const Level& level, Price p) {
                return side == Side::BUY ? level.price > p : level.price < p;
            });
        
        if (it != levels.end() && it->price == price) {
            it->prefetch();
            return &(*it);
        }
        
        Level new_level;
        new_level.price = price;
        new_level.orders.reserve(INITIAL_ORDER_CAPACITY / MAX_LEVELS);
        
        it = levels.insert(it, std::move(new_level));
        return &(*it);
    }
    
    Level* find_level(Price price, Side side) noexcept {
        auto& levels = side == Side::BUY ? bids_ : asks_;
        PREFETCH(levels.data());
        
        auto it = std::lower_bound(levels.begin(), levels.end(), price,
            [side](const Level& level, Price p) {
                return side == Side::BUY ? level.price > p : level.price < p;
            });
        
        if (it != levels.end() && it->price == price) {
            it->prefetch();
            return &(*it);
        }
        return nullptr;
    }

    void cleanup_levels(Side side) noexcept {
        auto& levels = side == Side::BUY ? bids_ : asks_;
        levels.erase(
            std::remove_if(levels.begin(), levels.end(),
                [](const Level& level) { return level.orders.empty(); }),
            levels.end());
    }
    
    template<typename Matcher>
    void match_order(Order& order, Matcher&& matcher) {
        auto& opposite_levels = order.side == Side::BUY ? asks_ : bids_;
        
        while (order.quantity > 0 && !opposite_levels.empty()) {
            Level& best_level = opposite_levels.front();
            best_level.prefetch();
            
            if ((order.side == Side::BUY && best_level.price > order.price) ||
                (order.side == Side::SELL && best_level.price < order.price)) {
                break;
            }
            
            for (auto it = best_level.orders.begin(); it != best_level.orders.end() && order.quantity > 0; ) {
                Order* resting_order = *it;
                PREFETCH(resting_order);
                
                Quantity trade_qty = std::min(order.quantity, resting_order->quantity - resting_order->filled_quantity);
                
                if (trade_qty > 0) {
                    matcher(order, *resting_order, best_level.price, trade_qty);
                    
                    order.quantity -= trade_qty;
                    order.filled_quantity += trade_qty;
                    
                    resting_order->filled_quantity += trade_qty;
                    best_level.total_quantity.fetch_sub(trade_qty, std::memory_order_relaxed);
                    
                    if (resting_order->filled_quantity == resting_order->quantity) {
                        resting_order->status.store(OrderStatus::FILLED, std::memory_order_release);
                        it = best_level.orders.erase(it);
                        continue;
                    } else {
                        resting_order->status.store(OrderStatus::PARTIALLY_FILLED, std::memory_order_release);
                    }
                    
                    if (order.quantity == 0) {
                        order.status.store(OrderStatus::FILLED, std::memory_order_release);
                        break;
                    }
                }
                ++it;
            }
            
            if (best_level.orders.empty()) {
                opposite_levels.erase(opposite_levels.begin());
            } else {
                break;
            }
        }
    }

public:
    Orderbook() {
        bids_.reserve(MAX_LEVELS);
        asks_.reserve(MAX_LEVELS);
    }
    
    Orderbook(const Orderbook&) = delete;
    Orderbook& operator=(const Orderbook&) = delete;
    
    OrderId add_order(Side side, OrderType type, Price price, Quantity quantity) {
        if (quantity <= 0 || (type != OrderType::MARKET && price <= 0)) {
            return 0;
        }
        
        OrderId id = generate_id();
        auto order = std::make_unique<Order>(id, price, quantity, side, type, get_timestamp());
        Order* order_ptr = order.get();
        orders_map_.emplace(id, std::move(order));
        
        if (type == OrderType::MARKET || type == OrderType::IOC) {
            match_order(*order_ptr, [](Order&, Order&, Price p, Quantity q) {
                std::cout << "Trade: " << q << " @ " << p << "\n";
            });
            
            if (order_ptr->quantity > 0 && type == OrderType::IOC) {
                order_ptr->status.store(OrderStatus::CANCELLED, std::memory_order_release);
            } else if (order_ptr->quantity == 0) {
                order_ptr->status.store(OrderStatus::FILLED, std::memory_order_release);
            }
        } else if (order_ptr->quantity > 0) {
            Level* level = find_or_create_level(price, side);
            level->orders.push_back(order_ptr);
            level->total_quantity.fetch_add(order_ptr->quantity, std::memory_order_relaxed);
            
            if (level->orders.size() > 1) {
                std::sort(level->orders.begin(), level->orders.end(),
                    [](const Order* a, const Order* b) { return a->timestamp < b->timestamp; });
            }
        }
        
        return id;
    }
    
    bool modify_order(OrderId id, Price new_price, Quantity new_quantity) {
        auto it = orders_map_.find(id);
        if (it == orders_map_.end()) return false;
        
        Order* order = it->second.get();
        OrderStatus status = order->status.load(std::memory_order_acquire);
        
        if (status != OrderStatus::OPEN && status != OrderStatus::PARTIALLY_FILLED) {
            return false;
        }
        
        if (new_quantity <= 0 || (order->type != OrderType::MARKET && new_price <= 0)) {
            return false;
        }
        
        Side side = order->side;
        OrderType type = order->type;
        Quantity remaining_qty = order->quantity - order->filled_quantity;
        
        if (!cancel_order(id)) return false;
        if (remaining_qty > 0) add_order(side, type, new_price, new_quantity);
        return true;
    }
    
    bool cancel_order(OrderId id) {
        auto it = orders_map_.find(id);
        if (it == orders_map_.end()) return false;
        
        Order* order = it->second.get();
        OrderStatus expected = OrderStatus::OPEN;
        
        if (!order->status.compare_exchange_strong(expected, OrderStatus::CANCELLED,
                                                 std::memory_order_release,
                                                 std::memory_order_relaxed)) {
            return false;
        }
        
        if (order->type == OrderType::LIMIT) {
            Level* level = find_level(order->price, order->side);
            if (level) {
                auto& orders = level->orders;
                orders.erase(std::remove(orders.begin(), orders.end(), order), orders.end());
                level->total_quantity.fetch_sub(order->quantity - order->filled_quantity, 
                                              std::memory_order_relaxed);
                cleanup_levels(order->side);
            }
        }
        
        order->quantity = order->filled_quantity;
        return true;
    }
    
    std::pair<Price, Price> get_top() const noexcept {
        return {
            bids_.empty() ? 0 : bids_.front().price,
            asks_.empty() ? 0 : asks_.front().price
        };
    }
    
    OrderStatus get_order_status(OrderId id) const noexcept {
        auto it = orders_map_.find(id);
        return it == orders_map_.end() ? OrderStatus::REJECTED 
                                      : it->second->status.load(std::memory_order_acquire);
    }
    
    Quantity get_volume_at(Price price, Side side) const noexcept {
        const auto& levels = side == Side::BUY ? bids_ : asks_;
        auto it = std::lower_bound(levels.begin(), levels.end(), price,
            [side](const Level& level, Price p) {
                return side == Side::BUY ? level.price > p : level.price < p;
            });
        
        return (it != levels.end() && it->price == price) 
            ? it->total_quantity.load(std::memory_order_relaxed) : 0;
    }
    
    void print_book() const {
        std::cout << "Bids:\n";
        for (const auto& level : bids_) {
            std::cout << "  " << level.price << " (" << level.total_quantity << ")\n";
        }
        std::cout << "Asks:\n";
        for (const auto& level : asks_) {
            std::cout << "  " << level.price << " (" << level.total_quantity << ")\n";
        }
    }
};

} // namespace Orderbook

int main() {
    Orderbook::Orderbook book;
    
    // Test orders with usage to avoid unused variable warnings
    auto bid1 = book.add_order(Orderbook::Side::BUY, Orderbook::OrderType::LIMIT, 100, 500);
    auto bid2 = book.add_order(Orderbook::Side::BUY, Orderbook::OrderType::LIMIT, 99, 300);
    auto ask1 = book.add_order(Orderbook::Side::SELL, Orderbook::OrderType::LIMIT, 101, 400);
    auto ask2 = book.add_order(Orderbook::Side::SELL, Orderbook::OrderType::LIMIT, 102, 200);
    
    std::cout << "Initial book:\n";
    book.print_book();
    std::cout << "Order statuses:\n";
    std::cout << "bid1: " << static_cast<int>(book.get_order_status(bid1)) << "\n";
    std::cout << "bid2: " << static_cast<int>(book.get_order_status(bid2)) << "\n";
    std::cout << "ask1: " << static_cast<int>(book.get_order_status(ask1)) << "\n";
    std::cout << "ask2: " << static_cast<int>(book.get_order_status(ask2)) << "\n";
    
    auto market_buy = book.add_order(Orderbook::Side::BUY, Orderbook::OrderType::MARKET, 0, 350);
    std::cout << "\nAfter market buy:\n";
    book.print_book();
    std::cout << "market_buy status: " << static_cast<int>(book.get_order_status(market_buy)) << "\n";
    
    auto crossing_sell = book.add_order(Orderbook::Side::SELL, Orderbook::OrderType::LIMIT, 99, 200);
    std::cout << "\nAfter crossing sell:\n";
    book.print_book();
    std::cout << "crossing_sell status: " << static_cast<int>(book.get_order_status(crossing_sell)) << "\n";
    
    if (book.cancel_order(bid2)) {
        std::cout << "\nCancelled bid2, new status: " 
                  << static_cast<int>(book.get_order_status(bid2)) << "\n";
    }
    
    std::cout << "\nFinal book:\n";
    book.print_book();
    
    return 0;
}