// ./build/bin/mlir-opt --decompose-linalg-softmax --one-shot-bufferize="bufferize-function-boundaries" -convert-linalg-to-affine-loops -convert-kvcacheupdate-to-affine-loops test/Conversion/TorchToTensor/gpt_full_prompt.mlir
module {
  func.func @qkt_computaton(%Wq: tensor<1x1024x1024xf32>, %Wk: tensor<1x1024x1024xf32>, %Wv: tensor<1x1024x1024xf32>, %x: tensor<1x256x1024xf32> ,%K_cache: tensor<1x16x512x64xf32>, %V_cache: tensor<1x16x512x64xf32>, %used_cache: index, %new_index: index) -> tensor<1x16x256x64xf32> {
    // Obtain Q and K by projecting Wq and Wk
    %initQ= tensor.empty() : tensor<1x256x1024xf32>
    %Q = linalg.batch_matmul
      ins(%x, %Wq : tensor<1x256x1024xf32>, tensor<1x1024x1024xf32>)
      outs(%initQ : tensor<1x256x1024xf32>)
      -> tensor<1x256x1024xf32>
    
    %newShape = arith.constant dense<[1, 16, 256, 64]> : tensor<4xi64>
    %Q_reshaped = tensor.reshape %Q(%newShape)
    : (tensor<1x256x1024xf32>, tensor<4xi64>) -> tensor<1x16x256x64xf32>

    %initK = tensor.empty() : tensor<1x256x1024xf32>
    %K = linalg.batch_matmul
      ins(%x, %Wk : tensor<1x256x1024xf32>, tensor<1x1024x1024xf32>)
      outs(%initK : tensor<1x256x1024xf32>)
      -> tensor<1x256x1024xf32>
    
    %K_reshaped = tensor.reshape %K(%newShape)
    : (tensor<1x256x1024xf32>, tensor<4xi64>) -> tensor<1x16x256x64xf32>

    %initV = tensor.empty() : tensor<1x256x1024xf32>
    %V = linalg.batch_matmul
      ins(%x, %Wv : tensor<1x256x1024xf32>, tensor<1x1024x1024xf32>)
      outs(%initV : tensor<1x256x1024xf32>)
      -> tensor<1x256x1024xf32>
    
    %V_reshaped = tensor.reshape %V(%newShape)
    : (tensor<1x256x1024xf32>, tensor<4xi64>) -> tensor<1x16x256x64xf32>


    // Insert K into K_cache at offset [0, iteration_id, 0]
    %updated_K_cache = linalg.kvcacheupdate
        %K_reshaped into %K_cache
        %used_cache
        [0, 0, 0, 0]          // offsets (dyn, static)
        [1, 16, 256, 64]  // sizes   (dyn, static)
        [1, 1, 1, 1]       // strides (dyn, static)
        : tensor<1x16x256x64xf32> into tensor<1x16x512x64xf32>
          -> tensor<1x16x512x64xf32>
    %copied_initK = tensor.empty() : tensor<1x16x512x64xf32>
    %copied_Kcache = linalg.copy
      ins(%K_cache : tensor<1x16x512x64xf32>)
      outs(%copied_initK : tensor<1x16x512x64xf32>)
      -> tensor<1x16x512x64xf32>
    
    %zero = arith.constant 0 : index
    // Apply mask to K_cache
    %result = linalg.kvcachemask
        %copied_Kcache : tensor<1x16x512x64xf32>
        starting_point(%new_index)
        mask_value(%zero)
        {dimension = 3 : i64 }
        -> tensor<1x16x512x64xf32>
    
    // Insert V into V_cache at offset [0, iteration_id, 0]
    %updated_V_cache = linalg.kvcacheupdate
        %V_reshaped into %V_cache
        %used_cache
        [0, 0, 0, 0]          // offsets (dyn, static)
        [1, 16, 256, 64]  // sizes   (dyn, static)
        [1, 1, 1, 1]       // strides (dyn, static)
        : tensor<1x16x256x64xf32> into tensor<1x16x512x64xf32>
          -> tensor<1x16x512x64xf32>
    
    %copied_initV = tensor.empty() : tensor<1x16x512x64xf32>
    %copied_Vcache = linalg.copy
      ins(%V_cache : tensor<1x16x512x64xf32>)
      outs(%copied_initV : tensor<1x16x512x64xf32>)
      -> tensor<1x16x512x64xf32>
    
    %masked_vcache = linalg.kvcachemask
        %copied_Vcache : tensor<1x16x512x64xf32>
        starting_point(%new_index)
        mask_value(%zero)
        {dimension = 2 : i64 }
        -> tensor<1x16x512x64xf32>

    %initK_t = tensor.empty() : tensor<1x16x64x512xf32>

    // Transpose K
    %K_T = linalg.transpose
      ins(%result : tensor<1x16x512x64xf32>)
      outs(%initK_t : tensor<1x16x64x512xf32>)
      permutation = [0, 1, 3, 2]
    
    //Prepare output tensor for matmul
    %initScore = tensor.empty() : tensor<1x16x256x512xf32>

    %scores = linalg.generic
    { indexing_maps = [
        affine_map<(b, h, m, k, n) -> (b, h, m, k)>, // A
        affine_map<(b, h, m, k, n) -> (b, h, k, n)>, // B
        affine_map<(b, h, m, k, n) -> (b, h, m, n)>  // Output
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]
    }
    ins(%Q_reshaped, %K_T : tensor<1x16x256x64xf32>, tensor<1x16x64x512xf32>)
    outs(%initScore : tensor<1x16x256x512xf32>) {
        ^bb0(%a: f32, %b: f32, %out: f32):
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %out, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<1x16x256x512xf32>
    
    %ten = arith.constant 10 : index
    %masked_result = linalg.kvcachemask
        %scores : tensor<1x16x256x512xf32>
        starting_point(%new_index)
        mask_value(%ten)
        { dimension = 3 : i64 }
        -> tensor<1x16x256x512xf32>

    
     %initSoftmax = tensor.empty() : tensor<1x16x256x512xf32>
    %softmaxed = linalg.softmax dimension(2) ins(%masked_result : tensor<1x16x256x512xf32>) outs(%initSoftmax : tensor<1x16x256x512xf32>) -> tensor<1x16x256x512xf32>


    %masked_softmax = linalg.kvcachemask
        %softmaxed : tensor<1x16x256x512xf32>
        starting_point(%new_index)
        mask_value(%zero)
        { dimension = 3 : i64 }
        -> tensor<1x16x256x512xf32>
    %initOutput = tensor.empty() : tensor<1x16x256x64xf32>
    %Output = linalg.generic
    { indexing_maps = [
        affine_map<(b, h, m, k, n) -> (b, h, m, k)>, // A
        affine_map<(b, h, m, k, n) -> (b, h, k, n)>, // B
        affine_map<(b, h, m, k, n) -> (b, h, m, n)>  // Output
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]
    }
    ins(%masked_softmax, %masked_vcache : tensor<1x16x256x512xf32>, tensor<1x16x512x64xf32>)
    outs(%initOutput : tensor<1x16x256x64xf32>) {
        ^bb0(%a: f32, %b: f32, %out: f32):
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %out, %prod : f32
        linalg.yield %sum : f32
    } -> tensor<1x16x256x64xf32>


  return %Output : tensor<1x16x256x64xf32>
  }
}