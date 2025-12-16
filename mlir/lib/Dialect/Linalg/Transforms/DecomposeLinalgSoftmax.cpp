//===- ElementwiseToLinalg.cpp - conversion of elementwise to linalg ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/Support/Debug.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define DEBUG_TYPE "softmax-decomposition-pass"

namespace mlir {
#define GEN_PASS_DEF_DECOMPOSELINALGSOFTMAXPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// Pattern to decompose linalg.softmax into primitive ops: exp, sum, div
/*
Softmax op decomposition: 
1. First all the ops have to be exponentiated using linalg.map with math.exp
2. Then the sum has to be computed along the softmax dimension using linalg.reduce
3. Finally, the division has to be performed using linalg.map again
*/
struct DecomposeSoftmaxPattern : public OpRewritePattern<linalg::SoftmaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value output = op.getOutput();
    RankedTensorType inputType = dyn_cast<RankedTensorType>(input.getType());
    assert(inputType && "expected ranked tensor type");
    int64_t rank = inputType.getRank();

    // The softmax dimension
    int64_t dim = op.getDimension();

    // Step 1: exp(x)
    LLVM_DEBUG(llvm::dbgs() << " Performing exponentation phase \n");
    // whats the expoenentation size: ? same as the input size
    auto expTensorType = inputType;
    // rewriter.setInsertionPointToStart(op->getBlock());
    rewriter.setInsertionPointAfter(op);
    Value expInit = rewriter.create<tensor::EmptyOp>(loc, expTensorType.getShape(),expTensorType.getElementType());
    auto expOp = rewriter.create<linalg::MapOp>(
    loc,
    /*inputs=*/ValueRange{input},
    /*init=*/expInit,
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value inVal = args[0];
      Value expVal = b.create<math::ExpOp>(loc, inVal);
      b.create<linalg::YieldOp>(loc, expVal);
    });
    LLVM_DEBUG(llvm::dbgs() << " Exponentiation done \n");

    // Now second step: sum along the softmax dimension
    LLVM_DEBUG(llvm::dbgs() << " Performing summation phase \n");
    // The sum tensor type will have one less dimension along 'dim'
    SmallVector<int64_t, 4> sumShape;
    for (int64_t i = 0; i < rank; ++i) {
      if (i != dim)
        sumShape.push_back(inputType.getDimSize(i));
    }
    auto sumTensorType = RankedTensorType::get(sumShape, inputType.getElementType());
    Value sumInit = rewriter.create<tensor::EmptyOp>(loc, sumTensorType.getShape(),sumTensorType.getElementType());
    Value zeroConst = rewriter.create<arith::ConstantOp>(
    loc, rewriter.getZeroAttr(inputType.getElementType()));
    // Initialize the sum tensor to zeros
    auto initFillOp = rewriter.create<linalg::FillOp>(
    loc,
    /*input=*/zeroConst,
    /*output=*/sumInit);
    LLVM_DEBUG(llvm::dbgs()<<"The fill op is: \n");
    LLVM_DEBUG(initFillOp.print(llvm::dbgs()));  
    LLVM_DEBUG(llvm::dbgs() << " Initialization done \n");
    auto sumOp = rewriter.create<linalg::ReduceOp>(
    loc,
    /*inputs=*/ValueRange{expOp.getResult()},
    /*init=*/initFillOp.getResult(0),
    /*dimensions=*/ArrayRef<int64_t>{dim},
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value a = args[0];
      Value bVal = args[1];
      Value sum = b.create<arith::AddFOp>(loc, a, bVal);
      b.create<linalg::YieldOp>(loc, sum);
    });
    LLVM_DEBUG(llvm::dbgs()<<"The sum op is: \n");
    LLVM_DEBUG(sumOp.print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << " Summation done \n");
    // Now we broadcast the sum result to match the exp tensor shape
    SmallVector<int64_t, 4> broadcastShape;
    for (int64_t i = 0; i < rank; ++i) {
      if (i == dim)
        broadcastShape.push_back(inputType.getDimSize(i));
      else
        broadcastShape.push_back(sumTensorType.getDimSize(i < dim ? i : i - 1));
    }
    auto broadcastTensorType = RankedTensorType::get(broadcastShape, inputType.getElementType());
    Value broadcastedSum = rewriter.create<tensor::EmptyOp>(loc, broadcastShape,broadcastTensorType.getElementType());
    auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
    loc,
    /*input=*/sumOp.getResult(0),
    /*output=*/broadcastedSum, dim);
    LLVM_DEBUG(llvm::dbgs()<<"The broadcast op is: \n");
    LLVM_DEBUG(broadcastOp.print(llvm::dbgs()));  
    LLVM_DEBUG(llvm::dbgs() << " Broadcasting done \n");
    // Finally, step 4: divide exp(x) by sum
    LLVM_DEBUG(llvm::dbgs() << " Performing division phase \n");
    SmallVector<Value, 2> divOpInputs;
    divOpInputs.push_back(expOp.getResult()[0]);
    divOpInputs.push_back(broadcastOp.getResult()[0]);

    Value finalInit = rewriter.create<tensor::EmptyOp>(loc, inputType.getShape(),inputType.getElementType());
    auto divOp = rewriter.create<linalg::MapOp>(
    loc,
    /*inputs=*/divOpInputs,
    /*init=*/finalInit,
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value expVal = args[0];
      Value sumVal = args[1];
      Value divVal = b.create<arith::DivFOp>(loc, expVal, sumVal);
      b.create<linalg::YieldOp>(loc, divVal);
    });
    LLVM_DEBUG(llvm::dbgs()<<"The division op is: \n");
    LLVM_DEBUG(divOp.print(llvm::dbgs()));  
    rewriter.replaceAllUsesWith(op.getResult(), divOp.getResult());
    LLVM_DEBUG(llvm::dbgs() <<"Replaced the oop \n");
    rewriter.eraseOp(op);
    LLVM_DEBUG(llvm::dbgs() << " Softmax decomposition completed \n");
    return success();
  }
};

class DecomposeLinalgSoftmaxPass
    : public impl::DecomposeLinalgSoftmaxPassBase<DecomposeLinalgSoftmaxPass> {
  using Base::Base;

  void runOnOperation() final {
    auto *func = getOperation();
    auto *context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    // Mark everything legal except linalg::SoftmaxOp
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           tensor::TensorDialect>();
    target.addIllegalOp<linalg::SoftmaxOp>();

    patterns.add<DecomposeSoftmaxPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createDecomposeLinalgSoftmaxPass() {
  return std::make_unique<DecomposeLinalgSoftmaxPass>();
}