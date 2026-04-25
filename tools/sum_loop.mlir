module {
  tt.func @sum_0_to_10(%0: !tt.ptr<i32>) {
    %1 = "arith.constant"() {value = 0 : i32} : () -> i32
    %2 = "arith.constant"() {value = 1 : i32} : () -> i32
    %3 = "arith.constant"() {value = 10 : i32} : () -> i32
    %7 = "scf.for"(%1, %3, %2, %1) ({
      ^bb0(%4: i32, %5: i32):
      %6 = "arith.addi"(%5, %4) : (i32, i32) -> i32
      "scf.yield"(%6) : (i32) -> ()
    }) : (i32, i32, i32, i32) -> i32
    "tt.store"(%0, %7) {operandSegmentSizes = array<i32: 1, 1, 0>} : (!tt.ptr<i32>, i32) -> ()
    "tt.return"() : () -> ()
  }
}
