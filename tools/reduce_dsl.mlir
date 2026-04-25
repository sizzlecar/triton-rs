module {
  tt.func @sum_block(%0: !tt.ptr<f32>) {
    %1 = "arith.constant"() {value = 0.0 : f32} : () -> f32
    %2 = "tt.splat"(%1) : (f32) -> tensor<128xf32>
    %6 = "tt.reduce"(%2) ({
      ^bb0(%3: f32, %4: f32):
      %5 = "arith.addf"(%3, %4) : (f32, f32) -> f32
      "tt.reduce.return"(%5) : (f32) -> ()
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    "tt.store"(%0, %6) {operandSegmentSizes = array<i32: 1, 1, 0>} : (!tt.ptr<f32>, f32) -> ()
    "tt.return"() : () -> ()
  }
}
