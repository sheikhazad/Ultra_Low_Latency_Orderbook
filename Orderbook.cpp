// ====================================================================== //
// Ultra Low Latency Orderbook - Optimized Version                        //
// Key Features:                                                         //
//   - Cache-line aligned structures for false sharing avoidance          //
//   - Price-time priority matching engine                               //
//   - Lock-free order ID generation                                     //
//   - Platform-specific optimizations (prefetching, timestamps)         //
// ====================================================================== //

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
#include <unordered_map> //Can use tsl::robin_map for faster processing

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
    #define OS_WINDOWS 1
    #include <intrin.h>
#elif defined(__linux__)
    #define OS_LINUX 1
    #include <x86intrin.h>
#elif defined(__APPLE__)
    #define OS_MAC 1
    #include <libkern/OSAtomic.h>
#else
    #define OS_UNKNOWN 1
#endif

// Performance Macros
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    #define PREFETCH(addr) __builtin_prefetch(addr)
    #define PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1)
#elif defined(_MSC_VER)
    #define PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
    #define PREFETCH_WRITE(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T1)
#else
    #define PREFETCH(addr) ((void)0)
    #define PREFETCH_WRITE(addr) ((void)0)
#endif

// Cache line size
#ifndef hardware_destructive_interference_size
    #define hardware_destructive_interference_size 64
#endif

namespace Orderbook {

// Type Aliases
using Price = int64_t;          // Price in ticks
using Quantity = int64_t;       // Quantity in lots
using OrderId = uint64_t;       // Unique order identifier
using Timestamp = uint64_t;     // High-resolution timestamp

// Configuration Constants
constexpr size_t CACHE_LINE_SIZE = hardware_destructive_interference_size;
constexpr size_t MAX_LEVELS = 1024;
constexpr size_t INITIAL_ORDER_CAPACITY = 4096;
constexpr Price INVALID_PRICE = 0;  // Marker for invalid prices

// Enums
enum class OrderType : uint8_t { LIMIT = 0, MARKET = 1, IOC = 2 };
enum class Side : uint8_t { BUY = 0, SELL = 1 };
enum class OrderStatus : uint8_t { OPEN = 0, PARTIALLY_FILLED = 1, FILLED = 2, CANCELLED = 3, REJECTED = 4 };

// ========== Type Aliases ========== //
using Price = int64_t;          // Price in ticks (avoid floating point)
using Quantity = int64_t;       // Quantity in lots
using OrderId = uint64_t;       // Unique order identifier
using Timestamp = uint64_t;     // High-resolution timestamp

// ========== AlignedAllocator ========== //
/**
 * Cache-line aligned allocator to prevent false sharing.
 * Uses platform-specific aligned allocation functions.
 */
template<typename T, size_t Alignment = CACHE_LINE_SIZE>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using size_type = std::size_t;

    template<typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    /**
     * Allocates aligned memory block
     * @param n Number of elements to allocate
     * @return Pointer to allocated memory
     * @throws std::bad_alloc on failure
     */
    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }

        void* ptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T))) {
            throw std::bad_alloc();
        }
#endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

// Aligned vector type for cache-friendly storage
template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

// ========== Order ========== //
/**
 * Order representation with cache-line alignment.
 * Contains all order metadata and execution state.
 */
struct alignas(CACHE_LINE_SIZE) Order {
    OrderId id;                     // Unique order ID
    Price price;                    // Limit price (0 for market orders)
    Quantity quantity;              // Original quantity
    Quantity filled_quantity;       // Executed quantity
    Side side;                      // BUY or SELL
    OrderType type;                 // Order type
    std::atomic<OrderStatus> status;// Atomic status for lock-free checks
    Timestamp timestamp;            // Time of order entry (for FIFO)

    Order(OrderId i, Price p, Quantity q, Side s, OrderType t, Timestamp ts)
        : id(i), price(p), quantity(q), filled_quantity(0), side(s), type(t), 
          status(OrderStatus::OPEN), timestamp(ts) {}
    
    // Non-copyable but movable
    Order(const Order&) = delete;
    Order& operator=(const Order&) = delete;
    Order(Order&&) = default;
    Order& operator=(Order&&) = default;
    
    /**
     * Price-time priority comparator
     * @param other Order to compare against
     * @return true if this order has higher priority
     */
    bool operator<(const Order& other) const noexcept {
        if (side == Side::BUY) {
            // Buy orders: higher price first, then older timestamp
            return price > other.price || (price == other.price && timestamp < other.timestamp);
        }
        // Sell orders: lower price first, then older timestamp
        return price < other.price || (price == other.price && timestamp < other.timestamp);
    }
};

// ========== Level ========== //
/**
 * Price level containing all orders at a specific price.
 * Maintains total quantity for quick volume checks.
 */
struct alignas(CACHE_LINE_SIZE) Level {
    Price price;                    // Price point
    AlignedVector<Order*> orders;   // Orders at this price (time-ordered)
    std::atomic<Quantity> total_quantity;  // Total volume at level
    
    Level() : price(INVALID_PRICE), total_quantity(0) {}
    
    // Move operations for efficient book updates
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
    
    /**
     * Prefetches level data for low-latency access
     */
    void prefetch() const noexcept {
        PREFETCH(this);
        if (!orders.empty()) PREFETCH(orders.data());
    }
};

// ========== Orderbook ========== //
class Orderbook {
private:
    // Thread-safe ID generation (atomic increment)
    alignas(CACHE_LINE_SIZE) std::atomic<OrderId> sequence_id_{0};
    
    // Order storage (consider robin_map for faster lookups)
    alignas(CACHE_LINE_SIZE) std::unordered_map<OrderId, std::unique_ptr<Order>> orders_map_;
    
    // Bid/ask levels (sorted vectors for cache locality)
    // Aligned vectors (Remove alignas, allocator handles it)
    /*alignas(CACHE_LINE_SIZE)*/AlignedVector<Level> bids_;
    AlignedVector<Level> asks_;

    /**
     * Generates monotonic order IDs
     * @return New unique order ID
     */
    OrderId generate_id() noexcept {
        return sequence_id_.fetch_add(1, std::memory_order_relaxed);
    }
    
    /**
     * Gets high-resolution timestamp
     * Uses CPU cycle counter when available, falls back to system clock
     * @return Current timestamp
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
     * Finds or creates a price level
     * @param price Price level to locate
     * @param side BUY or SELL
     * @return Pointer to existing or new level
     */
    Level* find_or_create_level(Price price, Side side) {
        auto& levels = side == Side::BUY ? bids_ : asks_;
        PREFETCH(levels.data());
        
        // Binary search for price level
        auto it = std::lower_bound(levels.begin(), levels.end(), price,
            [side](const Level& level, Price p) {
                return side == Side::BUY ? level.price > p : level.price < p;
            });
        
        // Return existing level if found
        if (it != levels.end() && it->price == price) {
            it->prefetch();
            return &(*it);
        }
        
        // Insert new level with reserved capacity
        Level new_level;
        new_level.price = price;
        new_level.orders.reserve(INITIAL_ORDER_CAPACITY / MAX_LEVELS);
        
        it = levels.insert(it, std::move(new_level));
        return &(*it);
    }
    
    /**
     * Locates price level without creation
     * @param price Price level to find
     * @param side BUY or SELL
     * @return Pointer to level or nullptr
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
     * Removes empty price levels
     * @param side BUY or SELL side to clean
     */
    void cleanup_levels(Side side) noexcept {
        auto& levels = side == Side::BUY ? bids_ : asks_;
        levels.erase(
            std::remove_if(levels.begin(), levels.end(),
                [](const Level& level) { return level.orders.empty(); }),
            levels.end());
    }
    
    /**
     * Matches incoming order against resting orders
     * @tparam Matcher Functor to handle trade execution
     * @param order Incoming order to match
     * @param matcher Trade execution callback
     */
    template<typename Matcher>
    void match_order(Order& order, Matcher&& matcher) {
        auto& opposite_levels = order.side == Side::BUY ? asks_ : bids_;
        
        while (order.quantity > 0 && !opposite_levels.empty()) {
            Level& best_level = opposite_levels.front();
            best_level.prefetch();
            
            // Price check for limit orders
            if ((order.side == Side::BUY && best_level.price > order.price) ||
                (order.side == Side::SELL && best_level.price < order.price)) {
                break;
            }
            
            // Match against orders at this price level
            for (auto it = best_level.orders.begin(); it != best_level.orders.end() && order.quantity > 0; ) {
                Order* resting_order = *it;
                PREFETCH(resting_order);
                
                // Calculate executable quantity
                Quantity trade_qty = std::min(order.quantity, resting_order->quantity - resting_order->filled_quantity);
                
                if (trade_qty > 0) {
                    // Execute trade
                    matcher(order, *resting_order, best_level.price, trade_qty);
                    
                    // Update order states
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
                    
                    // Check if incoming order is fully filled
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
     * Adds new order to the book
     * @param side BUY or SELL
     * @param type LIMIT, MARKET, or IOC
     * @param price Limit price (0 for market orders)
     * @param quantity Order quantity
     * @return Order ID or 0 if rejected
     */
    OrderId add_order(Side side, OrderType type, Price price, Quantity quantity) {
        // Validate parameters
        if (quantity <= 0 || (type != OrderType::MARKET && price <= 0)) {
            return 0;
        }
        
        // Create new order
        OrderId id = generate_id();
        auto order = std::make_unique<Order>(id, price, quantity, side, type, get_timestamp());
        Order* order_ptr = order.get();
        orders_map_.emplace(id, std::move(order));
        
        // Handle immediate execution orders
        if (type == OrderType::MARKET || type == OrderType::IOC) {
            match_order(*order_ptr, [](Order&, Order&, Price p, Quantity q) {
                std::cout << "Trade: " << q << " @ " << p << "\n";
            });
            
            // Handle IOC remaining quantity
            if (order_ptr->quantity > 0 && type == OrderType::IOC) {
                order_ptr->status.store(OrderStatus::CANCELLED, std::memory_order_release);
            } else if (order_ptr->quantity == 0) {
                order_ptr->status.store(OrderStatus::FILLED, std::memory_order_release);
            }
        } 
        // Add resting limit order
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
     * Modifies existing order (cancel + replace)
     * @param id Order ID to modify
     * @param new_price New limit price
     * @param new_quantity New quantity
     * @return True if modification succeeded
     */
    bool modify_order(OrderId id, Price new_price, Quantity new_quantity) {
        auto it = orders_map_.find(id);
        if (it == orders_map_.end()) return false;
        
        Order* order = it->second.get();
        OrderStatus status = order->status.load(std::memory_order_acquire);
        
        // Only modify open or partially filled orders
        if (status != OrderStatus::OPEN && status != OrderStatus::PARTIALLY_FILLED) {
            return false;
        }
        
        // Validate new parameters
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
     * Cancels an order
     * @param id Order ID to cancel
     * @return True if cancellation succeeded
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
        
        // Remove from price level if limit order
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
     * Gets best bid/ask prices
     * @return Pair of (best_bid, best_ask), 0 if empty
     */
    std::pair<Price, Price> get_top() const noexcept {
        return {
            bids_.empty() ? INVALID_PRICE : bids_.front().price,
            asks_.empty() ? INVALID_PRICE : asks_.front().price
        };
    }
    
    /**
     * Gets order status
     * @param id Order ID to check
     * @return Current order status
     */
    OrderStatus get_order_status(OrderId id) const noexcept {
        auto it = orders_map_.find(id);
        return it == orders_map_.end() ? OrderStatus::REJECTED 
                                      : it->second->status.load(std::memory_order_acquire);
    }
    
    /**
     * Gets total volume at price level
     * @param price Price level to check
     * @param side BUY or SELL side
     * @return Total quantity at price level, 0 if not found
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
     * Prints current orderbook state
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

int main() {
    Orderbook::Orderbook book;
    
    // Test scenario 1: Basic limit orders
    (void)book.add_order(Orderbook::Side::BUY, Orderbook::OrderType::LIMIT, 100, 500);
    auto bid2 = book.add_order(Orderbook::Side::BUY, Orderbook::OrderType::LIMIT, 99, 300);
    (void)book.add_order(Orderbook::Side::SELL, Orderbook::OrderType::LIMIT, 101, 400);
    (void)book.add_order(Orderbook::Side::SELL, Orderbook::OrderType::LIMIT, 102, 200);
    
    std::cout << "Initial book:\n";
    book.print_book();
    
    // Test scenario 2: Market order execution
    (void)book.add_order(Orderbook::Side::BUY, Orderbook::OrderType::MARKET, 0, 350);
    std::cout << "\nAfter market buy:\n";
    book.print_book();
    
    // Test scenario 3: Aggressive limit order
    (void)book.add_order(Orderbook::Side::SELL, Orderbook::OrderType::LIMIT, 99, 200);
    std::cout << "\nAfter crossing sell:\n";
    book.print_book();
    
    // Test scenario 4: Order cancellation
    if (book.cancel_order(bid2)) {
        std::cout << "\nCancelled bid2\n";
    }
    
    std::cout << "\nFinal book:\n";
    book.print_book();
    
    return 0;
}
