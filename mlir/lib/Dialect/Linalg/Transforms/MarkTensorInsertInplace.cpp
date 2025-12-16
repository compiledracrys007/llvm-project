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

namespace mlir {
#define GEN_PASS_DEF_MARKTENSORINSERTINPLACEPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {


class MarkTensorInsertInplacePass
    : public impl::MarkTensorInsertInplacePassBase<MarkTensorInsertInplacePass> {
  using Base::Base;

  void runOnOperation() final {
    auto *context = &getContext();
    getOperation()->walk([&](tensor::ExtractSliceOp extractOp) {
      extractOp->setAttr("cache_fetch", UnitAttr::get(context));
    });
    getOperation()->walk([&](tensor::InsertSliceOp insertOp) {
      insertOp->setAttr("cache_update", UnitAttr::get(context));
    });
  }
};

} // namespace

std::unique_ptr<Pass> createMarkTensorInsertInplacePass() {
  return std::make_unique<MarkTensorInsertInplacePass>();
}