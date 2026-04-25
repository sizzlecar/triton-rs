module {
  tt.func @sum_squares(%0: !tt.ptr<i32>) {
    %1 = "arith.constant"() {value = 0 : i32} : () -> i32
    %2 = "arith.constant"() {value = 10 : i32} : () -> i32
    %3 = "arith.constant"() {value = 1 : i32} : () -> i32
    %4 = "arith.constant"() {value = 0 : i32} : () -> i32
    %9 = "scf.for"(%1, %2, %3, %4) ({
      ^bb0(%5: i32, %6: i32):
      %7 = "arith.muli"(%5, %5) : (i32, i32) -> i32
      %8 = "arith.addi"(%6, %7) : (i32, i32) -> i32
      "scf.yield"(%8) : (i32) -> ()
    }) : (i32, i32, i32, i32) -> i32
    "tt.store"(%0, %9) {operandSegmentSizes = array<i32: 1, 1, 0>} : (!tt.ptr<i32>, i32) -> ()
    "tt.return"() : () -> ()
  }
}
