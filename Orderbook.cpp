// Ultra Low Latency Orderbook 
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

// Platform detection macros
#if defined(_WIN32) || defined(_WIN64)
    #define OS_WINDOWS 1
#elif defined(__linux__)
    #define OS_LINUX 1
#elif defined(__APPLE__)
    #define OS_MAC 1
#else
    #define OS_UNKNOWN 1
#endif

// Prefetch macros for performance optimization (x86/ARM/MSVC)
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

// Cache line size (64 bytes for most modern CPUs)
#ifndef hardware_destructive_interference_size
    #if defined(_MSC_VER)
        #define hardware_destructive_interference_size 64
    #else
        #define hardware_destructive_interference_size 64
    #endif
#endif

namespace Orderbook {

// Constants for orderbook configuration
constexpr size_t CACHE_LINE_SIZE = hardware_destructive_interference_size;
constexpr size_t MAX_LEVELS = 1024;          // Max price levels in the book
constexpr size_t INITIAL_ORDER_CAPACITY = 4096;  // Initial order capacity per level

// Order types, sides, and statuses
enum class OrderType : uint8_t { LIMIT = 0, MARKET = 1, IOC = 2 };
enum class Side : uint8_t { BUY = 0, SELL = 1 };
enum class OrderStatus : uint8_t { OPEN = 0, PARTIALLY_FILLED = 1, FILLED = 2, CANCELLED = 3, REJECTED = 4 };

// Type aliases for readability
using Price = int64_t;
using Quantity = int64_t;
using OrderId = uint64_t;
using Timestamp = uint64_t;

// ========== AlignedAllocator ========== //
/**
 * Custom allocator to ensure memory alignment for cache-line optimization.
 * @tparam T Data type to allocate.
 * @tparam Alignment Alignment size (default: CACHE_LINE_SIZE).
 */
template<typename T, size_t Alignment = CACHE_LINE_SIZE>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using size_type = std::size_t;

    template<typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    // Allocates aligned memory
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

    // Deallocates memory
    void deallocate(pointer p, size_type) noexcept {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    // Constructs an object in aligned memory
    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        new((void*)p) U(std::forward<Args>(args)...);
    }

    // Destroys an object
    template<typename U>
    void destroy(U* p) {
        p->~U();
    }
};

// Aligned vector for cache-friendly storage
template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

// ========== Order ========== //
/**
 * Represents an order in the orderbook.
 * Aligned to cache line to avoid false sharing.
 */
struct alignas(CACHE_LINE_SIZE) Order {
    OrderId id;                     // Unique order ID
    Price price;                    // Order price (0 for MARKET orders)
    Quantity quantity;              // Original quantity
    Quantity filled_quantity;       // Filled quantity
    Side side;                      // BUY or SELL
    OrderType type;                 // LIMIT, MARKET, or IOC
    std::atomic<OrderStatus> status;// Atomic order status
    Timestamp timestamp;            // Order timestamp (for FIFO matching)

    Order(OrderId i, Price p, Quantity q, Side s, OrderType t, Timestamp ts)
        : id(i), price(p), quantity(q), filled_quantity(0), side(s), type(t), 
          status(OrderStatus::OPEN), timestamp(ts) {}
    
    // Delete copy semantics (orders are non-copyable)
    Order(const Order&) = delete;
    Order& operator=(const Order&) = delete;
    Order(Order&&) = default;
    Order& operator=(Order&&) = default;
    
    // Comparator for price-time priority
    bool operator<(const Order& other) const noexcept {
        if (side == Side::BUY) {
            return price > other.price || (price == other.price && timestamp < other.timestamp);
        }
        return price < other.price || (price == other.price && timestamp < other.timestamp);
    }
};

// ========== Level ========== //
/**
 * Represents a price level in the orderbook.
 * Contains all orders at a specific price.
 */
struct alignas(CACHE_LINE_SIZE) Level {
    Price price;                    // Price level
    AlignedVector<Order*> orders;   // Orders at this level
    std::atomic<Quantity> total_quantity;  // Total quantity at this level
    
    Level() : price(0), total_quantity(0) {}
    
    // Move semantics for performance
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
    
    // Prefetches data for low-latency access
    void prefetch() const noexcept {
        PREFETCH(this);
        if (!orders.empty()) PREFETCH(orders.data());
    }
};

// ========== Orderbook ========== //
class Orderbook {
private:
    // Thread-safe order ID generation
    alignas(CACHE_LINE_SIZE) std::atomic<OrderId> sequence_id_{0};
    
    // Order storage (hash map for O(1) access)
    alignas(CACHE_LINE_SIZE) std::unordered_map<OrderId, std::unique_ptr<Order>> orders_map_;
    
    // Bid/ask levels (sorted vectors for cache locality)
    alignas(CACHE_LINE_SIZE) AlignedVector<Level> bids_;
    alignas(CACHE_LINE_SIZE) AlignedVector<Level> asks_;

    /**
     * Generates a unique order ID atomically.
     * @return New order ID.
     */
    OrderId generate_id() noexcept {
        return sequence_id_.fetch_add(1, std::memory_order_relaxed);
    }
    
    /**
     * Gets a high-resolution timestamp (CPU cycles or nanoseconds).
     * @return Timestamp for FIFO ordering.
     */
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
    
    /**
     * Finds or creates a price level for the given price and side.
     * @param price Price level to find/create.
     * @param side BUY or SELL.
     * @return Pointer to the Level.
     */
    Level* find_or_create_level(Price price, Side side) {
        auto& levels = side == Side::BUY ? bids_ : asks_;
        PREFETCH(levels.data());
        
        // Binary search for the price level
        auto it = std::lower_bound(levels.begin(), levels.end(), price,
            [side](const Level& level, Price p) {
                return side == Side::BUY ? level.price > p : level.price < p;
            });
        
        // Return existing level if found
        if (it != levels.end() && it->price == price) {
            it->prefetch();
            return &(*it);
        }
        
        // Insert new level
        Level new_level;
        new_level.price = price;
        new_level.orders.reserve(INITIAL_ORDER_CAPACITY / MAX_LEVELS);
        
        it = levels.insert(it, std::move(new_level));
        return &(*it);
    }
    
    /**
     * Finds a price level (returns nullptr if not found).
     * @param price Price level to find.
     * @param side BUY or SELL.
     * @return Pointer to the Level or nullptr.
     */
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

    /**
     * Removes empty price levels from the book.
     * @param side BUY or SELL.
     */
    void cleanup_levels(Side side) noexcept {
        auto& levels = side == Side::BUY ? bids_ : asks_;
        levels.erase(
            std::remove_if(levels.begin(), levels.end(),
                [](const Level& level) { return level.orders.empty(); }),
            levels.end());
    }
    
    /**
     * Matches an incoming order against resting orders.
     * @tparam Matcher Functor to handle trade execution.
     * @param order Incoming order to match.
     * @param matcher Callback for trade execution.
     */
    template<typename Matcher>
    void match_order(Order& order, Matcher&& matcher) {
        auto& opposite_levels = order.side == Side::BUY ? asks_ : bids_;
        
        while (order.quantity > 0 && !opposite_levels.empty()) {
            Level& best_level = opposite_levels.front();
            best_level.prefetch();
            
            // Check if order can match (price check)
            if ((order.side == Side::BUY && best_level.price > order.price) ||
                (order.side == Side::SELL && best_level.price < order.price)) {
                break;
            }
            
            // Match against resting orders
            for (auto it = best_level.orders.begin(); it != best_level.orders.end() && order.quantity > 0; ) {
                Order* resting_order = *it;
                PREFETCH(resting_order);
                
                Quantity trade_qty = std::min(order.quantity, resting_order->quantity - resting_order->filled_quantity);
                
                if (trade_qty > 0) {
                    // Execute trade via matcher callback
                    matcher(order, *resting_order, best_level.price, trade_qty);
                    
                    // Update quantities
                    order.quantity -= trade_qty;
                    order.filled_quantity += trade_qty;
                    
                    resting_order->filled_quantity += trade_qty;
                    best_level.total_quantity.fetch_sub(trade_qty, std::memory_order_relaxed);
                    
                    // Remove filled orders
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
            
            // Cleanup empty levels
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
    
    // Non-copyable
    Orderbook(const Orderbook&) = delete;
    Orderbook& operator=(const Orderbook&) = delete;
    
    /**
     * Adds a new order to the book.
     * @param side BUY or SELL.
     * @param type LIMIT, MARKET, or IOC.
     * @param price Order price (0 for MARKET).
     * @param quantity Order quantity.
     * @return Order ID (0 if rejected).
     */
    OrderId add_order(Side side, OrderType type, Price price, Quantity quantity) {
        if (quantity <= 0 || (type != OrderType::MARKET && price <= 0)) {
            return 0;  // Reject invalid orders
        }
        
        OrderId id = generate_id();
        auto order = std::make_unique<Order>(id, price, quantity, side, type, get_timestamp());
        Order* order_ptr = order.get();
        orders_map_.emplace(id, std::move(order));
        
        // Handle MARKET/IOC orders (immediate matching)
        if (type == OrderType::MARKET || type == OrderType::IOC) {
            match_order(*order_ptr, [](Order&, Order&, Price p, Quantity q) {
                std::cout << "Trade: " << q << " @ " << p << "\n";
            });
            
            // Cancel remaining IOC quantity
            if (order_ptr->quantity > 0 && type == OrderType::IOC) {
                order_ptr->status.store(OrderStatus::CANCELLED, std::memory_order_release);
            } else if (order_ptr->quantity == 0) {
                order_ptr->status.store(OrderStatus::FILLED, std::memory_order_release);
            }
        } 
        // Add LIMIT order to the book
        else if (order_ptr->quantity > 0) {
            Level* level = find_or_create_level(price, side);
            level->orders.push_back(order_ptr);
            level->total_quantity.fetch_add(order_ptr->quantity, std::memory_order_relaxed);
            
            // Maintain time priority
            if (level->orders.size() > 1) {
                std::sort(level->orders.begin(), level->orders.end(),
                    [](const Order* a, const Order* b) { return a->timestamp < b->timestamp; });
            }
        }
        
        return id;
    }
    
    /**
     * Modifies an existing order (cancel + replace).
     * @param id Order ID to modify.
     * @param new_price New price.
     * @param new_quantity New quantity.
     * @return True if successful.
     */
    bool modify_order(OrderId id, Price new_price, Quantity new_quantity) {
        auto it = orders_map_.find(id);
        if (it == orders_map_.end()) return false;
        
        Order* order = it->second.get();
        OrderStatus status = order->status.load(std::memory_order_acquire);
        
        // Only modify OPEN or PARTIALLY_FILLED orders
        if (status != OrderStatus::OPEN && status != OrderStatus::PARTIALLY_FILLED) {
            return false;
        }
        
        if (new_quantity <= 0 || (order->type != OrderType::MARKET && new_price <= 0)) {
            return false;
        }
        
        // Cancel and replace
        Side side = order->side;
        OrderType type = order->type;
        Quantity remaining_qty = order->quantity - order->filled_quantity;
        
        if (!cancel_order(id)) return false;
        if (remaining_qty > 0) add_order(side, type, new_price, new_quantity);
        return true;
    }
    
    /**
     * Cancels an order.
     * @param id Order ID to cancel.
     * @return True if successful.
     */
    bool cancel_order(OrderId id) {
        auto it = orders_map_.find(id);
        if (it == orders_map_.end()) return false;
        
        Order* order = it->second.get();
        OrderStatus expected = OrderStatus::OPEN;
        
        // Atomically cancel the order
        if (!order->status.compare_exchange_strong(expected, OrderStatus::CANCELLED,
                                                 std::memory_order_release,
                                                 std::memory_order_relaxed)) {
            return false;
        }
        
        // Remove from price level (if LIMIT order)
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
    
    /**
     * Gets the best bid/ask prices.
     * @return Pair of (best_bid, best_ask).
     */
    std::pair<Price, Price> get_top() const noexcept {
        return {
            bids_.empty() ? 0 : bids_.front().price,
            asks_.empty() ? 0 : asks_.front().price
        };
    }
    
    /**
     * Gets the status of an order.
     * @param id Order ID.
     * @return OrderStatus.
     */
    OrderStatus get_order_status(OrderId id) const noexcept {
        auto it = orders_map_.find(id);
        return it == orders_map_.end() ? OrderStatus::REJECTED 
                                      : it->second->status.load(std::memory_order_acquire);
    }
    
    /**
     * Gets the total volume at a price level.
     * @param price Price level.
     * @param side BUY or SELL.
     * @return Total quantity at the price.
     */
    Quantity get_volume_at(Price price, Side side) const noexcept {
        const auto& levels = side == Side::BUY ? bids_ : asks_;
        auto it = std::lower_bound(levels.begin(), levels.end(), price,
            [side](const Level& level, Price p) {
                return side == Side::BUY ? level.price > p : level.price < p;
            });
        
        return (it != levels.end() && it->price == price) 
            ? it->total_quantity.load(std::memory_order_relaxed) : 0;
    }
    
    /**
     * Prints the current state of the orderbook.
     */
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

// ========== Test Code ========== //
int main() {
    Orderbook::Orderbook book;
    
    // Test orders
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
    
    // Test market order
    auto market_buy = book.add_order(Orderbook::Side::BUY, Orderbook::OrderType::MARKET, 0, 350);
    std::cout << "\nAfter market buy:\n";
    book.print_book();
    std::cout << "market_buy status: " << static_cast<int>(book.get_order_status(market_buy)) << "\n";
    
    // Test crossing limit order
    auto crossing_sell = book.add_order(Orderbook::Side::SELL, Orderbook::OrderType::LIMIT, 99, 200);
    std::cout << "\nAfter crossing sell:\n";
    book.print_book();
    std::cout << "crossing_sell status: " << static_cast<int>(book.get_order_status(crossing_sell)) << "\n";
    
    // Test cancellation
    if (book.cancel_order(bid2)) {
        std::cout << "\nCancelled bid2, new status: " 
                  << static_cast<int>(book.get_order_status(bid2)) << "\n";
    }
    
    std::cout << "\nFinal book:\n";
    book.print_book();
    
    return 0;
}
