module {
  tt.func @vec_add(%0: !tt.ptr<f32>, %1: !tt.ptr<f32>, %2: !tt.ptr<f32>, %3: i32) {
    %4 = "tt.get_program_id"() {axis = 0 : i32} : () -> i32
    %5 = "arith.constant"() {value = 512 : i32} : () -> i32
    %6 = "arith.muli"(%4, %5) : (i32, i32) -> i32
    %7 = "tt.splat"(%6) : (i32) -> tensor<512xi32>
    %8 = "tt.make_range"() {start = 0 : i32, end = 512 : i32} : () -> tensor<512xi32>
    %9 = "arith.addi"(%7, %8) : (tensor<512xi32>, tensor<512xi32>) -> tensor<512xi32>
    %10 = "tt.splat"(%3) : (i32) -> tensor<512xi32>
    %11 = "arith.cmpi"(%9, %10) {predicate = 2 : i64} : (tensor<512xi32>, tensor<512xi32>) -> tensor<512xi1>
    %12 = "tt.splat"(%0) : (!tt.ptr<f32>) -> tensor<512x!tt.ptr<f32>>
    %13 = "tt.addptr"(%12, %9) : (tensor<512x!tt.ptr<f32>>, tensor<512xi32>) -> tensor<512x!tt.ptr<f32>>
    %14 = "tt.load"(%13, %11) {operandSegmentSizes = array<i32: 1, 1, 0>} : (tensor<512x!tt.ptr<f32>>, tensor<512xi1>) -> tensor<512xf32>
    %15 = "tt.splat"(%1) : (!tt.ptr<f32>) -> tensor<512x!tt.ptr<f32>>
    %16 = "tt.addptr"(%15, %9) : (tensor<512x!tt.ptr<f32>>, tensor<512xi32>) -> tensor<512x!tt.ptr<f32>>
    %17 = "tt.load"(%16, %11) {operandSegmentSizes = array<i32: 1, 1, 0>} : (tensor<512x!tt.ptr<f32>>, tensor<512xi1>) -> tensor<512xf32>
    %18 = "tt.splat"(%2) : (!tt.ptr<f32>) -> tensor<512x!tt.ptr<f32>>
    %19 = "tt.addptr"(%18, %9) : (tensor<512x!tt.ptr<f32>>, tensor<512xi32>) -> tensor<512x!tt.ptr<f32>>
    %20 = "arith.addf"(%14, %17) : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    "tt.store"(%19, %20, %11) {operandSegmentSizes = array<i32: 1, 1, 1>} : (tensor<512x!tt.ptr<f32>>, tensor<512xf32>, tensor<512xi1>) -> ()
    "tt.return"() : () -> ()
  }
}
