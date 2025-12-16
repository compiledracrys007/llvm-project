
#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define DEBUG_TYPE "kv-cache-pass"

namespace mlir {
#define GEN_PASS_DEF_CONVERTKVCACHEUPDATEOPTOAFFINELOOPSPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::affine;

namespace {

/// Pattern to convert KVCacheUpdateOp to affine loops.
// struct ConvertKVCacheUpdateOpToAffineLoopsPattern
//     : public OpRewritePattern<linalg::KVCacheUpdateOp> {
//   using OpRewritePattern<linalg::KVCacheUpdateOp>::OpRewritePattern;
//     LogicalResult matchAndRewrite(linalg::KVCacheUpdateOp op,
//                                     PatternRewriter &rewriter) const override {
//         Location loc = op.getLoc();
//         Value source = op.getSource();
//         Value dest = op.getDest();
//         // Print Debug info
//         LLVM_DEBUG(llvm::dbgs() << "KV Cache source " << source << "\n");
//         LLVM_DEBUG(llvm::dbgs() << "KV Cache dest " << dest << "\n");        
//         // Get the offsets, sizes and strides.
//         // Create affine loops to perform the copy from source to dest.
//         auto sourceType = dyn_cast<MemRefType>(source.getType());
//         auto destType = dyn_cast<MemRefType>(dest.getType());
//         assert(sourceType && destType && "Source and Dest must be MemRef types");
//         auto sourceShape = sourceType.getShape();
//         auto destShape = destType.getShape();
//         assert(sourceShape.size() == destShape.size() && "Source and Dest must have same rank");
//         SmallVector<int, 4> bounds;
//         for(auto sh : sourceShape) bounds.push_back(sh);
//         SmallVector<Value, 4> loopIvs;
//         AffineForOp outerLoop;
//         int cnt = 0, lb = 0;
//         AffineForOp forOp;
//         for (auto b :bounds) {
//             LLVM_DEBUG(llvm::dbgs() << "Bound: " << b << "\n");
//             if (cnt == 2) {
//                   Value usedC = op.getUsedCache();  // index-typed

//                   AffineMap lbMap =
//                       AffineMap::get(/*dimCount=*/1, 0,
//                                     rewriter.getAffineDimExpr(0)); // d0 → LB
                
//                 AffineMap ubMap = AffineMap::get(
//                     /*dimCount=*/1, 0,
//                     rewriter.getAffineDimExpr(0) +
//                     rewriter.getAffineConstantExpr(b));  // UB = d0 + b
                  

//                   forOp = rewriter.create<AffineForOp>(
//                       loc,
//                       /*lowerBoundOperands=*/ValueRange{usedC},
//                       lbMap,
//                       /*upperBoundOperands=*/ValueRange{usedC},
//                       ubMap);

//               } else {
//                 forOp = rewriter.create<AffineForOp>(loc, 0, b);
//             }
//             rewriter.setInsertionPointToStart(forOp.getBody());
//             if(cnt == 0) outerLoop = forOp;
//             loopIvs.push_back(forOp.getInductionVar());
//             // Inside the loop, perform the copy.
//             if(b == bounds.back()) {
//                 // Load from source and store to dest.
//                 Value load = rewriter.create<AffineLoadOp>(loc, source, loopIvs);
//                 rewriter.create<AffineStoreOp>(loc, load, dest, loopIvs);
//             }
//             cnt++;
//         }
//         rewriter.setInsertionPointAfter(outerLoop);
//         rewriter.replaceAllUsesWith(op, dest);
//         rewriter.eraseOp(op);
//         return success();
//     }
// };

struct ConvertKVCacheUpdateOpToAffineLoopsPattern
    : public OpRewritePattern<linalg::KVCacheUpdateOp> {
  using OpRewritePattern<linalg::KVCacheUpdateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::KVCacheUpdateOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = op.getSource();
    Value dest   = op.getDest();
    Value usedC  = op.getUsedCache(); // index

    auto sourceType = dyn_cast<MemRefType>(source.getType());
    auto destType   = dyn_cast<MemRefType>(dest.getType());
    assert(sourceType && destType && "Source/Dest must be memrefs");

    auto sourceShape = sourceType.getShape();
    assert(sourceShape.size() == 4 && "Expected rank-4 KV cache");

    SmallVector<Value, 4> loopIvs;
    AffineForOp outerLoop;

    for (int i = 0; i < sourceShape.size(); i++) {
      int64_t bound = sourceShape[i];
      auto forOp = rewriter.create<AffineForOp>(loc, 0, bound);

      if (i == 0)
        outerLoop = forOp;

      rewriter.setInsertionPointToStart(forOp.getBody());
      loopIvs.push_back(forOp.getInductionVar());
    }

    // === Load / Store ===
    SmallVector<Value, 4> srcIdxs;
    SmallVector<Value, 4> dstIdxs;

    for (int i = 0; i < loopIvs.size(); i++) {
      Value iv = loopIvs[i];
      srcIdxs.push_back(iv);

      if (i == 2) {
        // Shift only the cache dimension
        AffineExpr d0 = rewriter.getAffineDimExpr(0);
AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
AffineMap map = AffineMap::get(/*dims=*/1, /*symbols=*/1, d0 + s0);

Value shifted = rewriter.create<affine::AffineApplyOp>(
    loc, map, ValueRange{iv, usedC});

dstIdxs.push_back(shifted);

      } else {
        dstIdxs.push_back(iv);
      }
    }

    Value val =
        rewriter.create<AffineLoadOp>(loc, source, srcIdxs);
    rewriter.create<AffineStoreOp>(loc, val, dest, dstIdxs);

    rewriter.setInsertionPointAfter(outerLoop);
    rewriter.replaceAllUsesWith(op, dest);
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertKVCacheMaskOpToAffineLoopsPattern
    : public OpRewritePattern<linalg::KVCacheMaskOp> {
  using OpRewritePattern<linalg::KVCacheMaskOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(linalg::KVCacheMaskOp op,
                                    PatternRewriter &rewriter) const override {
        // Implementation would go here.
        Location loc = op.getLoc();
        Value source = op.getSource();
        Value mask_val = op.getMaskValue();
        Value starting_point = op.getStartingPoint();
        auto dimension = op.getDimension();                                
        Value mask_f32;
        auto f32Ty = rewriter.getF32Type();

        if (auto cst = mask_val.getDefiningOp<arith::ConstantIndexOp>()) {
            // Statically known mask value
            int64_t iv = cst.value();

            float fv;
            if (iv > 2) {
                fv = -103.28;   // clamp in C++
            } else {
                fv = static_cast<float>(iv);
            }

            mask_f32 = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getF32FloatAttr(fv));

        } else {
            // Dynamic value — no comparison in IR
            Value mask_i32 = rewriter.create<arith::IndexCastOp>(
                loc, rewriter.getI32Type(), mask_val);
            mask_f32 = rewriter.create<arith::SIToFPOp>(
                loc, f32Ty, mask_i32);
        }
        LLVM_DEBUG(llvm::dbgs() << "KV Cache Mask value " << mask_f32 << "\n");
        LLVM_DEBUG(llvm::dbgs() << "KV Cache Mask dimension " << dimension << "\n");
        LLVM_DEBUG(llvm::dbgs() << "KV Cache Mask source " << source << "\n");

        auto sourceTy = dyn_cast<MemRefType>(source.getType());
        SmallVector<int, 4> bounds;
        auto sourceShape = sourceTy.getShape();
        for(auto sh : sourceShape) bounds.push_back(sh);
        SmallVector<Value, 4> loopIvs;
        AffineForOp outerLoop;
        int cnt = 0, lb = 0;
        AffineForOp forOp;
        for (auto b :bounds) {
            LLVM_DEBUG(llvm::dbgs() << "Bound: " << b << "\n");
            if (cnt == dimension) {
              LLVM_DEBUG(llvm::dbgs() << "Creating loop with starting point\n");
                AffineMap lbMap =AffineMap::get(/*dimCount=*/1, 0,
                                    rewriter.getAffineDimExpr(0)); // d0 → LB

                  AffineMap ubMap =
                      AffineMap::get(/*dimCount=*/0, 0,
                                    rewriter.getAffineConstantExpr(b)); // const UB

                  forOp = rewriter.create<AffineForOp>(
                      loc,
                      /*lowerBoundOperands=*/ValueRange{starting_point},
                      lbMap,
                      /*upperBoundOperands=*/ValueRange{},
                      ubMap);

              } else {
                forOp = rewriter.create<AffineForOp>(loc, 0, b);
            }
            rewriter.setInsertionPointToStart(forOp.getBody());
            if(cnt == 0) outerLoop = forOp;
            loopIvs.push_back(forOp.getInductionVar());
            // Inside the loop, perform the copy.
            if(b == bounds.back()) {
                // Load from source and store to dest.
                // Value load = rewriter.create<AffineLoadOp>(loc, source, loopIvs);
                rewriter.create<AffineStoreOp>(loc, mask_f32, source, loopIvs);
            }
            cnt++;
        }
        rewriter.setInsertionPointAfter(outerLoop);
        rewriter.replaceAllUsesWith(op, source);
        rewriter.eraseOp(op);
        return success(); // Placeholder
    }
};

class ConvertKVCacheUpdateOpToAffineLoopsPass
    : public impl::ConvertKVCacheUpdateOpToAffineLoopsPassBase<ConvertKVCacheUpdateOpToAffineLoopsPass> {
  using Base::Base;

  void runOnOperation() final {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertKVCacheUpdateOpToAffineLoopsPattern>(context);
    patterns.add<ConvertKVCacheMaskOpToAffineLoopsPattern>(context);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<Pass> createConvertKVCacheUpdateOpToAffineLoopsPass() {
  return std::make_unique<ConvertKVCacheUpdateOpToAffineLoopsPass>();
}