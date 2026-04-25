module {
  tt.func @matmul_one_block(%0: tensor<16x32xf16>, %1: tensor<32x16xf16>) {
    %2 = "arith.constant"() {value = 0.0 : f32} : () -> f32
    %3 = "tt.splat"(%2) : (f32) -> tensor<1xf32>
    %4 = "tt.expand_dims"(%3) {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1x1xf32>
    %5 = "tt.broadcast"(%4) : (tensor<1x1xf32>) -> tensor<16x16xf32>
    %6 = "tt.dot"(%0, %1, %5) : (tensor<16x32xf16>, tensor<32x16xf16>, tensor<16x16xf32>) -> tensor<16x16xf32>
    "tt.return"() : () -> ()
  }
}
