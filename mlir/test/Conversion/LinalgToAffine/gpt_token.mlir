// ./build/bin/mlir-opt --decompose-linalg-softmax --one-shot-bufferize="bufferize-function-boundaries" -convert-linalg-to-affine-loops -convert-kvcacheupdate-to-affine-loops mlir/test/Conversion/LinalgToAffine/gpt_token.mlir
module {
  func.func @token_computaton(
    %Wq: tensor<1x1024x1024xf32>,
    %Wk: tensor<1x1024x1024xf32>,
    %Wv: tensor<1x1024x1024xf32>,
    %x: tensor<1x1x1024xf32>,
    %K_cache: tensor<1x16x512x64xf32>,
    %V_cache: tensor<1x16x512x64xf32>,
    %used_cache: index,
    %new_index: index
  ) -> tensor<1x16x1x64xf32> {

    %zero_f32 = arith.constant 0.0 : f32
    %zero_idx = arith.constant 0 : index

    // ---- Q ----
    %initQ_empty = tensor.empty() : tensor<1x1x1024xf32>
    %initQ = linalg.fill ins(%zero_f32 : f32)
             outs(%initQ_empty : tensor<1x1x1024xf32>)
             -> tensor<1x1x1024xf32>

    %Q = linalg.batch_matmul
      ins(%x, %Wq : tensor<1x1x1024xf32>, tensor<1x1024x1024xf32>)
      outs(%initQ : tensor<1x1x1024xf32>)
      -> tensor<1x1x1024xf32>

    %newShape = arith.constant dense<[1, 16, 1, 64]> : tensor<4xi64>
    %Q_reshaped = tensor.reshape %Q(%newShape)
      : (tensor<1x1x1024xf32>, tensor<4xi64>)
      -> tensor<1x16x1x64xf32>

    // ---- K ----
    %initK_empty = tensor.empty() : tensor<1x1x1024xf32>
    %initK = linalg.fill ins(%zero_f32 : f32)
             outs(%initK_empty : tensor<1x1x1024xf32>)
             -> tensor<1x1x1024xf32>

    %K = linalg.batch_matmul
      ins(%x, %Wk : tensor<1x1x1024xf32>, tensor<1x1024x1024xf32>)
      outs(%initK : tensor<1x1x1024xf32>)
      -> tensor<1x1x1024xf32>

    %K_reshaped = tensor.reshape %K(%newShape)
      : (tensor<1x1x1024xf32>, tensor<4xi64>)
      -> tensor<1x16x1x64xf32>

    // ---- V ----
    %initV_empty = tensor.empty() : tensor<1x1x1024xf32>
    %initV = linalg.fill ins(%zero_f32 : f32)
             outs(%initV_empty : tensor<1x1x1024xf32>)
             -> tensor<1x1x1024xf32>

    %V = linalg.batch_matmul
      ins(%x, %Wv : tensor<1x1x1024xf32>, tensor<1x1024x1024xf32>)
      outs(%initV : tensor<1x1x1024xf32>)
      -> tensor<1x1x1024xf32>

    %V_reshaped = tensor.reshape %V(%newShape)
      : (tensor<1x1x1024xf32>, tensor<4xi64>)
      -> tensor<1x16x1x64xf32>

    // ---- Update K cache ----
    %updated_K_cache = linalg.kvcacheupdate
        %K_reshaped into %K_cache
        %used_cache
        [0, 0, 0, 0]
        [1, 16, 1, 64]
        [1, 1, 1, 1]
        : tensor<1x16x1x64xf32> into tensor<1x16x512x64xf32>
        -> tensor<1x16x512x64xf32>

    %copied_initK_empty = tensor.empty() : tensor<1x16x512x64xf32>
    %copied_initK = linalg.fill ins(%zero_f32 : f32)
                    outs(%copied_initK_empty : tensor<1x16x512x64xf32>)
                    -> tensor<1x16x512x64xf32>

    %copied_Kcache = linalg.copy
      ins(%K_cache : tensor<1x16x512x64xf32>)
      outs(%copied_initK : tensor<1x16x512x64xf32>)
      -> tensor<1x16x512x64xf32>

    %masked_Kcache = linalg.kvcachemask
        %copied_Kcache : tensor<1x16x512x64xf32>
        starting_point(%new_index)
        mask_value(%zero_idx)
        { dimension = 2 : i64 }
        -> tensor<1x16x512x64xf32>

    // ---- Update V cache ----
    %updated_V_cache = linalg.kvcacheupdate
        %V_reshaped into %V_cache
        %used_cache
        [0, 0, 0, 0]
        [1, 16, 1, 64]
        [1, 1, 1, 1]
        : tensor<1x16x1x64xf32> into tensor<1x16x512x64xf32>
        -> tensor<1x16x512x64xf32>

    %copied_initV_empty = tensor.empty() : tensor<1x16x512x64xf32>
    %copied_initV = linalg.fill ins(%zero_f32 : f32)
                    outs(%copied_initV_empty : tensor<1x16x512x64xf32>)
                    -> tensor<1x16x512x64xf32>

    %copied_Vcache = linalg.copy
      ins(%V_cache : tensor<1x16x512x64xf32>)
      outs(%copied_initV : tensor<1x16x512x64xf32>)
      -> tensor<1x16x512x64xf32>

    %masked_Vcache = linalg.kvcachemask
        %copied_Vcache : tensor<1x16x512x64xf32>
        starting_point(%new_index)
        mask_value(%zero_idx)
        { dimension = 2 : i64 }
        -> tensor<1x16x512x64xf32>

    // ---- Transpose K ----
    %initK_t_empty = tensor.empty() : tensor<1x16x64x512xf32>
    %initK_t = linalg.fill ins(%zero_f32 : f32)
                outs(%initK_t_empty : tensor<1x16x64x512xf32>)
                -> tensor<1x16x64x512xf32>

    %K_T = linalg.transpose
      ins(%masked_Kcache : tensor<1x16x512x64xf32>)
      outs(%initK_t : tensor<1x16x64x512xf32>)
      permutation = [0, 1, 3, 2]

    // ---- Attention scores ----
    %initScore_empty = tensor.empty() : tensor<1x16x1x512xf32>
    %initScore = linalg.fill ins(%zero_f32 : f32)
                  outs(%initScore_empty : tensor<1x16x1x512xf32>)
                  -> tensor<1x16x1x512xf32>

    %scores = linalg.generic
      { indexing_maps = [
          affine_map<(b, h, m, k, n) -> (b, h, m, k)>,
          affine_map<(b, h, m, k, n) -> (b, h, k, n)>,
          affine_map<(b, h, m, k, n) -> (b, h, m, n)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]
      }
      ins(%Q_reshaped, %K_T : tensor<1x16x1x64xf32>, tensor<1x16x64x512xf32>)
      outs(%initScore : tensor<1x16x1x512xf32>) {
        ^bb0(%a: f32, %b: f32, %out: f32):
          %prod = arith.mulf %a, %b : f32
          %sum  = arith.addf %out, %prod : f32
          linalg.yield %sum : f32
      } -> tensor<1x16x1x512xf32>

    // ---- Softmax ----
    %initSoftmax_empty = tensor.empty() : tensor<1x16x1x512xf32>
    %initSoftmax = linalg.fill ins(%zero_f32 : f32)
                    outs(%initSoftmax_empty : tensor<1x16x1x512xf32>)
                    -> tensor<1x16x1x512xf32>
    %ten = arith.constant 10 : index
    %masked_result = linalg.kvcachemask
        %scores : tensor<1x16x1x512xf32>
        starting_point(%new_index)
        mask_value(%ten)
        { dimension = 3 : i64 }
        -> tensor<1x16x1x512xf32>

    %softmaxed = linalg.softmax dimension(3)
      ins(%masked_result : tensor<1x16x1x512xf32>)
      outs(%initSoftmax : tensor<1x16x1x512xf32>)
      -> tensor<1x16x1x512xf32>

    %masked_softmax = linalg.kvcachemask
        %softmaxed : tensor<1x16x1x512xf32>
        starting_point(%new_index)
        mask_value(%zero_idx)
        { dimension = 3 : i64 }
        -> tensor<1x16x1x512xf32>

    // ---- Final output ----
    %initOutput_empty = tensor.empty() : tensor<1x16x1x64xf32>
    %initOutput = linalg.fill ins(%zero_f32 : f32)
                   outs(%initOutput_empty : tensor<1x16x1x64xf32>)
                   -> tensor<1x16x1x64xf32>

    %Output = linalg.generic
      { indexing_maps = [
          affine_map<(b, h, m, k, n) -> (b, h, m, k)>,
          affine_map<(b, h, m, k, n) -> (b, h, k, n)>,
          affine_map<(b, h, m, k, n) -> (b, h, m, n)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]
      }
      ins(%masked_softmax, %masked_Vcache
          : tensor<1x16x1x512xf32>, tensor<1x16x512x64xf32>)
      outs(%initOutput : tensor<1x16x1x64xf32>) {
        ^bb0(%a: f32, %b: f32, %out: f32):
          %prod = arith.mulf %a, %b : f32
          %sum  = arith.addf %out, %prod : f32
          linalg.yield %sum : f32
      } -> tensor<1x16x1x64xf32>

    return %Output : tensor<1x16x1x64xf32>
  }
}
