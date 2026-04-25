module {
  tt.func @vector_add(%0: !tt.ptr<f32>, %1: !tt.ptr<f32>, %2: !tt.ptr<f32>, %3: i32) {
    %4 = "tt.get_program_id"() {axis = 0 : i32} : () -> i32
    %5 = "arith.constant"() {value = 1024 : i32} : () -> i32
    %6 = "arith.muli"(%4, %5) : (i32, i32) -> i32
    %7 = "tt.make_range"() {start = 0 : i32, end = 1024 : i32} : () -> tensor<1024xi32>
    %8 = "tt.splat"(%6) : (i32) -> tensor<1024xi32>
    %9 = "arith.addi"(%8, %7) : (tensor<1024xi32>, tensor<1024xi32>) -> tensor<1024xi32>
    %10 = "tt.splat"(%3) : (i32) -> tensor<1024xi32>
    %11 = "arith.cmpi"(%9, %10) {predicate = 2 : i64} : (tensor<1024xi32>, tensor<1024xi32>) -> tensor<1024xi1>
    %12 = "tt.splat"(%0) : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %13 = "tt.addptr"(%12, %9) : (tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) -> tensor<1024x!tt.ptr<f32>>
    %14 = "tt.load"(%13, %11) {operandSegmentSizes = array<i32: 1, 1, 0>} : (tensor<1024x!tt.ptr<f32>>, tensor<1024xi1>) -> tensor<1024xf32>
    %15 = "tt.splat"(%1) : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %16 = "tt.addptr"(%15, %9) : (tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) -> tensor<1024x!tt.ptr<f32>>
    %17 = "tt.load"(%16, %11) {operandSegmentSizes = array<i32: 1, 1, 0>} : (tensor<1024x!tt.ptr<f32>>, tensor<1024xi1>) -> tensor<1024xf32>
    %18 = "arith.addf"(%14, %17) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %19 = "tt.splat"(%2) : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %20 = "tt.addptr"(%19, %9) : (tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) -> tensor<1024x!tt.ptr<f32>>
    "tt.store"(%20, %18, %11) {operandSegmentSizes = array<i32: 1, 1, 1>} : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>, tensor<1024xi1>) -> ()
    "tt.return"() : () -> ()
  }
}
