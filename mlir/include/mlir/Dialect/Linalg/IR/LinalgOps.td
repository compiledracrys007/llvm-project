//===- LinalgOps.td - Linalg dialect ops -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for linear algebra operations.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_OPS
#define LINALG_OPS

include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"

// Base class for Linalg dialect ops that do not correspond to library calls.
class Linalg_Op<string mnemonic, list<Trait> traits = []> :
    Op<Linalg_Dialect, mnemonic, traits>;

def Linalg_YieldOp : Linalg_Op<"yield", [Pure, ReturnLike, Terminator]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Linalg yield operation";
  let description = [{
    `linalg.yield` is a special terminator operation for blocks inside regions
    in `linalg` generic ops. It returns values to the immediately enclosing
    `linalg` generic op.

    Example:

    ```mlir
    linalg.yield %f0, %f1 : f32, f32
    ```
  }];
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Linalg_IndexOp : Linalg_Op<"index", [Pure]>,
    Arguments<(ins ConfinedAttr<I64Attr, [IntMinValue<0>]>:$dim)>,
    Results<(outs Index:$result)> {
  let summary = "linalg index operation";
  let description = [{
    The `linalg.index` operation returns the iteration index of the immediately
    enclosing linalg structured operation for the iteration dimension `dim`. The
    `dim` attribute specifies the position of the accessed dimension in the
    indexing map domain.

    Example:

    ```mlir
    #map = affine_map<(i, j) -> (i, j)>
    linalg.generic {indexing_maps = [#map, #map],
                    iterator_types = ["parallel", "parallel"]}
      outs(%I, %J : memref<?x?xindex>, memref<?x?xindex>) {
      ^bb0(%arg0 : index, %arg1 : index):
      // Access the outer iteration dimension i
      %i = linalg.index 0 : index
      // Access the inner iteration dimension j
      %j = linalg.index 1 : index
      linalg.yield %i, %j : index, index
    }
    ```

    This may lower to IR resembling:

    ```mlir
    %0 = dim %I, %c0 : memref<?x?xindex>
    %1 = dim %I, %c1 : memref<?x?xindex>
    scf.for %i = %c0 to %0 step %c1 {
      scf.for %j = %c0 to %1 step %c1 {
        store %i, %I[%i, %j] : memref<?x?xindex>
        store %j, %J[%i, %j] : memref<?x?xindex>
      }
    }
    ```
  }];

  let assemblyFormat = [{ $dim attr-dict `:` type($result) }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def KVCacheUpdateOp : Linalg_Op<"kvcacheupdate", [AttrSizedOperandSegments, OffsetSizeAndStrideOpInterface,
  DeclareOpInterfaceMethods<BufferizableOpInterface>
]> {
  let summary = "KV Cache update operator";
  let description = [{
    Permutes the dimensions of `input` according to the given `permutation`.
      `dim(result, i) = dim(input, permutation[i])`

    This op actually moves data, unlike `memref.transpose` which is a metadata
    operation only that produces a transposed "view".

    Example:
    ```mlir
      %update = linalg.kv_cache_update
          ins(%input:tensor<16x64xf32>)
          outs(%init:tensor<64x16xf32>)
    ```
  }]; 

  let arguments = (ins
    // Input arg
    AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$source,
    AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$dest,
    Index:$used_cache,
    Variadic<Index>:$offsets, 
    Variadic<Index>:$strides, 
    Variadic<Index>:$sizes, 
    DenseI64ArrayAttr:$static_strides, 
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_offsets
  );
  let results = (outs AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$result);

  let assemblyFormat = [{
    $source `into` $dest ``
    $used_cache ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `into` type($dest) `->` type($result)
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultTy, "Value": $source, "Value": $dest, "Value":$used_cache,
    "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes, 
    "ArrayRef<OpFoldResult>":$strides, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>, 

    OpBuilder<(ins "Type":$resultTy, "Value":$source, "Value":$dest, "Value":$used_cache,
    "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange": $strides, 
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    
    OpBuilder<(ins "Type":$resultTy, "Value":$source, "Value":$dest,"Value":$used_cache,
      "ArrayRef<Range>":$ranges,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration= [{
    Type getSourceType(){
      return getSource().getType();
    }

    Type getDestType(){
      return getDest().getType();
    }
    
    std::array<unsigned, 3> getArrayAttrMaxRanks(){
      if(auto rankedTensorType = dyn_cast<RankedTensorType>(getDestType())){
        unsigned rank = rankedTensorType.getRank();
        return {rank, rank, rank};
      }
      if(auto memrefType = dyn_cast<MemRefType>(getDestType())){
        unsigned rank = memrefType.getRank();
        return {rank, rank, rank};
      }
      return {2, 2, 2};
    }

    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 3;}
    MutableOperandRange getDpsInitsMutable() { return getDestMutable();}

    SmallVector<OpFoldResult> getMixedOffsets(){
      OpBuilder b(getContext());
      return ::mlir::getMixedValues(getStaticOffsets(), getOffsets(), b);
    }
    SmallVector<OpFoldResult> getMixedSizes(){
      OpBuilder b(getContext());
      return ::mlir::getMixedValues(getStaticSizes(), getSizes(), b);
    }
    SmallVector<OpFoldResult> getMixedStrides(){
      OpBuilder b(getContext());
      return ::mlir::getMixedValues(getStaticStrides(), getStrides(), b);
    }
    bool bufferizesToMemoryRead(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return true;
    }
    bool bufferizesToMemoryWrite(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return false;
    }
    ::mlir::bufferization::AliasingValueList
    getAliasingValues(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return {};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state);
  }];
}

def KVCacheMaskOp : Linalg_Op<"kvcachemask", [
  DeclareOpInterfaceMethods<BufferizableOpInterface>
]> {
  let summary = "KV Cache mask operator";
  let description = [{
    Masks a copy of the kv cache with a given value

  }]; 

  let arguments = (ins
    // Input arg
    AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$source,
    Index:$starting_point,
    Index:$mask_value,
    I64Attr:$dimension
  );
  let results = (outs AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$result);

  let assemblyFormat = [{
  $source `:` type($source)
  `starting_point` `(` $starting_point `)`
  `mask_value` `(` $mask_value`)`
  attr-dict
  `->` type($result)
}];



  let extraClassDeclaration= [{
    Type getSourceType(){
      return getSource().getType();
    }
    bool bufferizesToMemoryRead(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return true;
    }
    bool bufferizesToMemoryWrite(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return true;
    }
    ::mlir::bufferization::AliasingValueList
    getAliasingValues(::mlir::OpOperand&,
        const ::mlir::bufferization::AnalysisState&) {
      return {};
    }
    LogicalResult bufferize(RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options,
                            bufferization::BufferizationState &state);
  }];
}


def Linalg_SoftmaxOp : Linalg_Op<"softmax",
    [DestinationStyleOpInterface,
     PredOpTrait<"input and output have same element type", TCopVTEtIsSameAs<0, 1>>,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
     DeclareOpInterfaceMethods<AggregatedOpInterface, ["decomposeOperation"]>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Softmax operator";
  let description = [{
    linalg.softmax computes a numerically stable version of softmax.

    For a given input tensor and a specified dimension `d`, compute:
      1. the max `m` along that dimension `d`
      2. f(x) = exp(x - m)
      3. sum f(x) along dimension d to get l(x).
      4. compute the final result f(x) / l(x).

    This is an aggregate linalg operation that further reduces to a small DAG of
    structured operations.

    Warning: Regarding the tiling capabilities, the implementation doesn't
    check that the provided dimensions make sense. This is the responsability
    of the transformation calling the tiling to ensure that the provided
    sizes for each dimension make sense with respect to the semantic of
    softmax.
  }];

  let arguments = (ins AnyShaped:$input,
                       AnyShaped:$output,
                       I64Attr:$dimension
  );

  let results = (outs Variadic<AnyRankedTensor>:$result);
  let hasFolder = 1;
  let assemblyFormat = [{
    attr-dict
    `dimension` `(` $dimension `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    ShapedType getInputOperandType() {
      return cast<ShapedType>(getInput().getType());
    }
    ShapedType getOutputOperandType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getInputOperandRank() {
      return getInputOperandType().getRank();
    }
    int64_t getOutputOperandRank() {
      return getOutputOperandType().getRank();
    }
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];
  let hasVerifier = 1;
}

def Linalg_WinogradFilterTransformOp : Linalg_Op<"winograd_filter_transform",
    [AllElementTypesMatch<["filter", "output"]>, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Winograd filter transform operator";
  let description = [{
    Winograd Conv2D algorithm will convert linalg Conv2D operator into batched
    matrix multiply. Before the matrix multiply, it will convert filter and
    input into a format suitable for batched matrix multiply. After the matrix
    multiply, it will convert output to the final result tensor.

    The algorithm F(m x m, r x r) is

    Y = A^T x [(G x g x G^T) @ (B^T x d x B)] x A

    The size of output Y is m x m. The size of filter g is r x r. The size of
    input d is (m + r - 1) x (m + r - 1). A^T, A, G^T, G, B^T, and B are
    transformation matrices.

    This operator is defined to represent the high level concept of filter
    transformation (G x g x G^T) in the Winograd Conv2D algorithm.
  }];

  let arguments = (ins TensorRankOf<[AnyType], [4]>:$filter,
                       TensorRankOf<[AnyType], [4]>:$output,
                       WinogradConv2DFmr:$fmr
  );

  let results = (outs TensorRankOf<[AnyType], [4]>:$result);
  let assemblyFormat = [{
    attr-dict
    `fmr` `(` $fmr `)`
    `ins` `(` $filter `:` type($filter) `)`
    `outs` `(` $output `:` type($output) `)`
    `->` type($result)
  }];
  let extraClassDeclaration = [{
    ShapedType getFilterOperandType() {
      return cast<ShapedType>(getFilter().getType());
    }
    ShapedType getOutputOperandType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getFilterOperandRank() {
      return getFilterOperandType().getRank();
    }
    int64_t getOutputOperandRank() {
      return getOutputOperandType().getRank();
    }
    int64_t getFilterFDim() {
      return 0;
    }
    int64_t getFilterHDim() {
      return 1;
    }
    int64_t getFilterWDim() {
      return 2;
    }
    int64_t getFilterCDim() {
      return 3;
    }
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];
  let hasVerifier = 1;
}

def Linalg_WinogradInputTransformOp : Linalg_Op<"winograd_input_transform",
    [AllElementTypesMatch<["input", "output"]>, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Winograd input transform operator";
  let description = [{
    Winograd Conv2D algorithm will convert linalg Conv2D operator into batched
    matrix multiply. Before the matrix multiply, it will convert filter and
    input into a format suitable for batched matrix multiply. After the matrix
    multiply, it will convert output to the final result tensor.

    The algorithm F(m x m, r x r) is

    Y = A^T x [(G x g x G^T) @ (B^T x d x B)] x A

    The size of output Y is m x m. The size of filter g is r x r. The size of
    input d is (m + r - 1) x (m + r - 1). A^T, A, G^T, G, B^T, and B are
    transformation matrices.

    This operator is defined to represent the high level concept of input
    transformation (B^T x d x B) in the Winograd Conv2D algorithm.
  }];

  let arguments = (ins TensorRankOf<[AnyType], [4]>:$input,
                       TensorRankOf<[AnyType], [6]>:$output,
                       WinogradConv2DFmr:$fmr
  );

  let results = (outs TensorRankOf<[AnyType], [6]>:$result);
  let assemblyFormat = [{
    attr-dict
    `fmr` `(` $fmr `)`
    `ins` `(` $input `:` type($input) `)`
    `outs` `(` $output `:` type($output) `)`
    `->` type($result)
  }];
  let extraClassDeclaration = [{
    ShapedType getInputOperandType() {
      return cast<ShapedType>(getInput().getType());
    }
    ShapedType getOutputOperandType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getInputOperandRank() {
      return getInputOperandType().getRank();
    }
    int64_t getOutputOperandRank() {
      return getOutputOperandType().getRank();
    }
    int64_t getInputNDim() {
      return 0;
    }
    int64_t getInputHDim() {
      return 1;
    }
    int64_t getInputWDim() {
      return 2;
    }
    int64_t getInputCDim() {
      return 3;
    }
    int64_t getOutputAlphaHDim() {
      return 0;
    }
    int64_t getOutputAlphaWDim() {
      return 1;
    }
    int64_t getOutputTileHDim() {
      return 2;
    }
    int64_t getOutputTileWDim() {
      return 3;
    }
    int64_t getOutputNDim() {
      return 4;
    }
    int64_t getOutputCDim() {
      return 5;
    }
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];
  let hasVerifier = 1;
}

def Linalg_WinogradOutputTransformOp : Linalg_Op<"winograd_output_transform",
    [AllElementTypesMatch<["value", "output"]>, DestinationStyleOpInterface,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation"]>]> {
  let summary = "Winograd output transform operator";
  let description = [{
    Winograd Conv2D algorithm will convert linalg Conv2D operator into batched
    matrix multiply. Before the matrix multiply, it will convert filter and
    input into a format suitable for batched matrix multiply. After the matrix
    multiply, it will convert output to the final result tensor.

    The algorithm F(m x m, r x r) is

    Y = A^T x [(G x g x G^T) @ (B^T x d x B)] x A

    The size of output Y is m x m. The size of filter g is r x r. The size of
    input d is (m + r - 1) x (m + r - 1). A^T, A, G^T, G, B^T, and B are
    transformation matrices.

    This operator is defined to represent the high level concept of output
    transformation (A^T x y x A) in the Winograd Conv2D algorithm.
  }];

  let arguments = (ins TensorRankOf<[AnyType], [6]>:$value,
                       TensorRankOf<[AnyType], [4]>:$output,
                       WinogradConv2DFmr:$fmr
  );

  let results = (outs TensorRankOf<[AnyType], [4]>:$result);
  let assemblyFormat = [{
    attr-dict
    `fmr` `(` $fmr `)`
    `ins` `(` $value `:` type($value) `)`
    `outs` `(` $output `:` type($output) `)`
    `->` type($result)
  }];
  let extraClassDeclaration = [{
    ShapedType getValueOperandType() {
      return cast<ShapedType>(getValue().getType());
    }
    ShapedType getOutputOperandType() {
      return cast<ShapedType>(getOutput().getType());
    }
    int64_t getValueOperandRank() {
      return getValueOperandType().getRank();
    }
    int64_t getOutputOperandRank() {
      return getOutputOperandType().getRank();
    }
    int64_t getValueAlphaHDim() {
      return 0;
    }
    int64_t getValueAlphaWDim() {
      return 1;
    }
    int64_t getValueTileHDim() {
      return 2;
    }
    int64_t getValueTileWDim() {
      return 3;
    }
    int64_t getValueNDim() {
      return 4;
    }
    int64_t getValueFDim() {
      return 5;
    }
    int64_t getOutputNDim() {
      return 0;
    }
    int64_t getOutputHDim() {
      return 1;
    }
    int64_t getOutputWDim() {
      return 2;
    }
    int64_t getOutputFDim() {
      return 3;
    }
    MutableOperandRange getDpsInitsMutable() { return getOutputMutable(); }
  }];
  let hasVerifier = 1;
}

#endif // LINALG_OPS
