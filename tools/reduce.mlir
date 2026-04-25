module {
  tt.func @sum_reduce_2d(%0: tensor<32x64xf32>) {
    %4 = "tt.reduce"(%0) ({
      ^bb0(%1: f32, %2: f32):
      %3 = "arith.addf"(%1, %2) : (f32, f32) -> f32
      "tt.reduce.return"(%3) : (f32) -> ()
    }) {axis = 1 : i32} : (tensor<32x64xf32>) -> tensor<32xf32>
    "tt.return"() : () -> ()
  }
}
