#!/bin/bash

# Comprehensive test runner for Spartan2 integration validation
# This script runs all tests to ensure the integration is successful

set -e  # Exit on any error

echo "🚀 Starting Spartan2 Integration Test Suite"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run tests with error handling
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    print_status "Running $test_name..."
    
    if eval "$test_command"; then
        print_success "$test_name passed"
        return 0
    else
        print_error "$test_name failed"
        return 1
    fi
}

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to update test counters
update_counters() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if [ $1 -eq 0 ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

echo ""
echo "📋 Test Plan:"
echo "1. Workspace Build and Unit Tests"
echo "2. Integration Tests"
echo "3. Performance Benchmarks"
echo "4. Edge Case Tests"
echo ""

# 1. Workspace Build and Unit Tests
print_status "Phase 1: Workspace Build and Unit Tests"
echo "========================================"

run_test "Workspace Build" "cargo build --workspace"
update_counters $?

run_test "Unit Tests" "cargo test --workspace"
update_counters $?

run_test "CCS Converter Tests" "cargo test -p neo-ccs converter_tests"
update_counters $?

run_test "Sum-check Tests" "cargo test -p neo-sumcheck spartan_sumcheck_tests"
update_counters $?

run_test "Folding Tests" "cargo test -p neo-fold spartan_folding_tests"
update_counters $?

run_test "Commitment Tests" "cargo test -p neo-commit spartan_pcs_tests"
update_counters $?

echo ""

# 2. Integration Tests
print_status "Phase 2: Integration Tests"
echo "=========================="

run_test "Integration Tests" "cargo test -p neo-orchestrator spartan_integration"
update_counters $?

run_test "Neo Main Demo" "cargo run -p neo-main"
update_counters $?

run_test "Neo Bench - Spartan2" "cargo run -p neo-bench -- --spartan2-fibo --min 3 --max 5"
update_counters $?

echo ""

# 4. Performance Benchmarks
print_status "Phase 4: Performance Benchmarks"
echo "================================"

print_warning "Performance benchmarks take longer to run..."

run_test "Performance Benchmark" "cargo bench -p neo-bench --bench spartan_performance"
update_counters $?

echo ""

# 5. Edge Case Tests
print_status "Phase 5: Edge Case and Error Handling Tests"
echo "============================================"

run_test "All Unit Tests" "cargo test --workspace"
update_counters $?

run_test "Documentation Tests" "cargo test --workspace --doc"
update_counters $?

echo ""

# 6. Comparative Analysis
print_status "Phase 6: Comparative Analysis"
echo "============================="

print_status "Running comparative benchmark between NARK and SNARK modes..."

# Run both modes and compare
if cargo run -p neo-bench -- --neo-fibo --spartan2-fibo --min 3 --max 4 > /tmp/comparison_results.txt 2>&1; then
    print_success "Comparative analysis completed"
    echo "Results saved to /tmp/comparison_results.txt"
    update_counters 0
else
    print_warning "Comparative analysis had issues (check /tmp/comparison_results.txt)"
    update_counters 1
fi

echo ""

# Final Results
echo "🏁 Test Suite Results"
echo "===================="
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"

if [ $FAILED_TESTS -eq 0 ]; then
    print_success "🎉 ALL TESTS PASSED! Spartan2 integration is successful!"
    echo ""
    echo "✅ Your Neo codebase now supports both NARK and SNARK modes"
    echo "✅ Backward compatibility is maintained"
    echo "✅ Spartan2 integration is working correctly"
    echo ""
    echo "Usage:"
    echo "  Demo: cargo run -p neo-main"
    echo "  Benchmarks: cargo run -p neo-bench -- --spartan2-fibo"
    echo "  Compare modes: cargo run -p neo-bench -- --neo-fibo --spartan2-fibo"
    exit 0
else
    print_error "❌ Some tests failed. Integration needs attention."
    echo ""
    echo "Failed tests: $FAILED_TESTS/$TOTAL_TESTS"
    echo "Please review the error messages above and fix the issues."
    exit 1
fi
