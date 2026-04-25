module {
  tt.func @hello_kernel(%0: !tt.ptr<i32>) {
    %1 = "tt.get_program_id"() {axis = 0 : i32} : () -> i32
    %2 = "arith.constant"() {value = 42 : i32} : () -> i32
    %3 = "arith.addi"(%1, %2) : (i32, i32) -> i32
    "tt.store"(%0, %3) {operandSegmentSizes = array<i32: 1, 1, 0>} : (!tt.ptr<i32>, i32) -> ()
    "tt.return"() : () -> ()
  }
}
