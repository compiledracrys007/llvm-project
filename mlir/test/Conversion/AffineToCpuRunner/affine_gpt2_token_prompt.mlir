// ./build/bin/mlir-opt -decompose-subview-ops -lower-affine -convert-scf-to-cf -convert-arith-to-llvm -convert-math-to-llvm -convert-cf-to-llvm --finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts mlir/test/Conversion/AffineToCpuRunner/affine_gpt2_token_prompt.mlir | ./build/bin/mlir-runner -entry-point-result=void -shared-libs=/Users/anirudhsathish/Codes/private/llvm-project/build/lib/libmlir_runner_utils.dylib,/Users/anirudhsathish/Codes/private/llvm-project/build/lib/libmlir_c_runner_utils.dylib
#map = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  func.func private @printMemrefF32(memref<*xf32>) attributes { llvm.emit_c_interface }
  memref.global "private" constant @__constant_4xi64 : memref<4xi64> = dense<[1, 16, 256, 64]> {alignment = 64 : i64}
  func.func @qkt_computaton(%arg0: memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>, %arg1: memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>, %arg2: memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>, %arg3: memref<1x256x1024xf32, strided<[?, ?, ?], offset: ?>>, %arg4: memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg5: memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg6: index, %arg7: index) -> memref<1x16x256x64xf32> {
    %cst = arith.constant -1.032800e+02 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_4xi64 : memref<4xi64>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x256x1024xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 256 {
        affine.for %arg10 = 0 to 1024 {
          affine.for %arg11 = 0 to 1024 {
            %1 = affine.load %arg3[%arg8, %arg9, %arg11] : memref<1x256x1024xf32, strided<[?, ?, ?], offset: ?>>
            %2 = affine.load %arg0[%arg8, %arg11, %arg10] : memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>
            %3 = affine.load %alloc[%arg8, %arg9, %arg10] : memref<1x256x1024xf32>
            %4 = arith.mulf %1, %2 : f32
            %5 = arith.addf %3, %4 : f32
            affine.store %5, %alloc[%arg8, %arg9, %arg10] : memref<1x256x1024xf32>
          }
        }
      }
    }
    %reshape = memref.reshape %alloc(%0) : (memref<1x256x1024xf32>, memref<4xi64>) -> memref<1x16x256x64xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x256x1024xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 256 {
        affine.for %arg10 = 0 to 1024 {
          affine.for %arg11 = 0 to 1024 {
            %1 = affine.load %arg3[%arg8, %arg9, %arg11] : memref<1x256x1024xf32, strided<[?, ?, ?], offset: ?>>
            %2 = affine.load %arg1[%arg8, %arg11, %arg10] : memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>
            %3 = affine.load %alloc_1[%arg8, %arg9, %arg10] : memref<1x256x1024xf32>
            %4 = arith.mulf %1, %2 : f32
            %5 = arith.addf %3, %4 : f32
            affine.store %5, %alloc_1[%arg8, %arg9, %arg10] : memref<1x256x1024xf32>
          }
        }
      }
    }
    %reshape_2 = memref.reshape %alloc_1(%0) : (memref<1x256x1024xf32>, memref<4xi64>) -> memref<1x16x256x64xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x256x1024xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 256 {
        affine.for %arg10 = 0 to 1024 {
          affine.for %arg11 = 0 to 1024 {
            %1 = affine.load %arg3[%arg8, %arg9, %arg11] : memref<1x256x1024xf32, strided<[?, ?, ?], offset: ?>>
            %2 = affine.load %arg2[%arg8, %arg11, %arg10] : memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>
            %3 = affine.load %alloc_3[%arg8, %arg9, %arg10] : memref<1x256x1024xf32>
            %4 = arith.mulf %1, %2 : f32
            %5 = arith.addf %3, %4 : f32
            affine.store %5, %alloc_3[%arg8, %arg9, %arg10] : memref<1x256x1024xf32>
          }
        }
      }
    }
    %reshape_4 = memref.reshape %alloc_3(%0) : (memref<1x256x1024xf32>, memref<4xi64>) -> memref<1x16x256x64xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = %arg6 to 256 {
          affine.for %arg11 = 0 to 64 {
            %1 = affine.load %reshape_2[%arg8, %arg9, %arg10, %arg11] : memref<1x16x256x64xf32>
            affine.store %1, %arg4[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>
          }
        }
      }
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x16x512x64xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 512 {
          affine.for %arg11 = 0 to 64 {
            %1 = affine.load %arg4[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>
            affine.store %1, %alloc_5[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32>
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 512 {
          affine.for %arg11 = %arg7 to 64 {
            affine.store %cst_0, %alloc_5[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32>
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = %arg6 to 256 {
          affine.for %arg11 = 0 to 64 {
            %1 = affine.load %reshape_4[%arg8, %arg9, %arg10, %arg11] : memref<1x16x256x64xf32>
            affine.store %1, %arg5[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>
          }
        }
      }
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x16x512x64xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 512 {
          affine.for %arg11 = 0 to 64 {
            %1 = affine.load %arg5[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>
            affine.store %1, %alloc_6[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32>
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = %arg7 to 512 {
          affine.for %arg11 = 0 to 64 {
            affine.store %cst_0, %alloc_6[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32>
          }
        }
      }
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x16x64x512xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 64 {
          affine.for %arg11 = 0 to 512 {
            %1 = affine.load %alloc_5[%arg8, %arg9, %arg11, %arg10] : memref<1x16x512x64xf32>
            affine.store %1, %alloc_7[%arg8, %arg9, %arg10, %arg11] : memref<1x16x64x512xf32>
          }
        }
      }
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x16x256x512xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 256 {
          affine.for %arg11 = 0 to 64 {
            affine.for %arg12 = 0 to 512 {
              %1 = affine.load %reshape[%arg8, %arg9, %arg10, %arg11] : memref<1x16x256x64xf32>
              %2 = affine.load %alloc_7[%arg8, %arg9, %arg11, %arg12] : memref<1x16x64x512xf32>
              %3 = affine.load %alloc_8[%arg8, %arg9, %arg10, %arg12] : memref<1x16x256x512xf32>
              %4 = arith.mulf %1, %2 : f32
              %5 = arith.addf %3, %4 : f32
              affine.store %5, %alloc_8[%arg8, %arg9, %arg10, %arg12] : memref<1x16x256x512xf32>
            }
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 256 {
          affine.for %arg11 = %arg7 to 512 {
            affine.store %cst, %alloc_8[%arg8, %arg9, %arg10, %arg11] : memref<1x16x256x512xf32>
          }
        }
      }
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x16x256x512xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 256 {
          affine.for %arg11 = 0 to 512 {
            %1 = affine.load %alloc_8[%arg8, %arg9, %arg10, %arg11] : memref<1x16x256x512xf32>
            %2 = math.exp %1 : f32
            affine.store %2, %alloc_9[%arg8, %arg9, %arg10, %arg11] : memref<1x16x256x512xf32>
          }
        }
      }
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x16x512xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 512 {
          affine.store %cst_0, %alloc_10[%arg8, %arg9, %arg10] : memref<1x16x512xf32>
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 256 {
          affine.for %arg11 = 0 to 512 {
            %1 = affine.load %alloc_9[%arg8, %arg9, %arg10, %arg11] : memref<1x16x256x512xf32>
            %2 = affine.load %alloc_10[%arg8, %arg9, %arg11] : memref<1x16x512xf32>
            %3 = arith.addf %1, %2 : f32
            affine.store %3, %alloc_10[%arg8, %arg9, %arg11] : memref<1x16x512xf32>
          }
        }
      }
    }
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x16x256x512xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 256 {
          affine.for %arg11 = 0 to 512 {
            %1 = affine.load %alloc_10[%arg8, %arg9, %arg11] : memref<1x16x512xf32>
            affine.store %1, %alloc_11[%arg8, %arg9, %arg10, %arg11] : memref<1x16x256x512xf32>
          }
        }
      }
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x16x256x512xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 256 {
          affine.for %arg11 = 0 to 512 {
            %1 = affine.load %alloc_9[%arg8, %arg9, %arg10, %arg11] : memref<1x16x256x512xf32>
            %2 = affine.load %alloc_11[%arg8, %arg9, %arg10, %arg11] : memref<1x16x256x512xf32>
            %3 = arith.divf %1, %2 : f32
            affine.store %3, %alloc_12[%arg8, %arg9, %arg10, %arg11] : memref<1x16x256x512xf32>
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 256 {
          affine.for %arg11 = %arg7 to 512 {
            affine.store %cst_0, %alloc_12[%arg8, %arg9, %arg10, %arg11] : memref<1x16x256x512xf32>
          }
        }
      }
    }
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x16x256x64xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 256 {
          affine.for %arg11 = 0 to 512 {
            affine.for %arg12 = 0 to 64 {
              %1 = affine.load %alloc_12[%arg8, %arg9, %arg10, %arg11] : memref<1x16x256x512xf32>
              %2 = affine.load %alloc_6[%arg8, %arg9, %arg11, %arg12] : memref<1x16x512x64xf32>
              %3 = affine.load %alloc_13[%arg8, %arg9, %arg10, %arg12] : memref<1x16x256x64xf32>
              %4 = arith.mulf %1, %2 : f32
              %5 = arith.addf %3, %4 : f32
              affine.store %5, %alloc_13[%arg8, %arg9, %arg10, %arg12] : memref<1x16x256x64xf32>
            }
          }
        }
      }
    }
    return %alloc_13 : memref<1x16x256x64xf32>
  }
  memref.global "private" constant @__constant_4 : memref<4xi64> = dense<[1, 16, 1, 64]> {alignment = 64 : i64}
  func.func @token_computaton(%arg0: memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>, %arg1: memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>, %arg2: memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>, %arg3: memref<1x1x1024xf32, strided<[?, ?, ?], offset: ?>>, %arg4: memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg5: memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg6: index, %arg7: index) -> memref<1x16x1x64xf32> {
    %cst = arith.constant -1.032800e+02 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_4 : memref<4xi64>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x1024xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 1024 {
          affine.store %cst_0, %alloc[%arg8, %arg9, %arg10] : memref<1x1x1024xf32>
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 1024 {
          affine.for %arg11 = 0 to 1024 {
            %1 = affine.load %arg3[%arg8, %arg9, %arg11] : memref<1x1x1024xf32, strided<[?, ?, ?], offset: ?>>
            %2 = affine.load %arg0[%arg8, %arg11, %arg10] : memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>
            %3 = affine.load %alloc[%arg8, %arg9, %arg10] : memref<1x1x1024xf32>
            %4 = arith.mulf %1, %2 : f32
            %5 = arith.addf %3, %4 : f32
            affine.store %5, %alloc[%arg8, %arg9, %arg10] : memref<1x1x1024xf32>
          }
        }
      }
    }
    %reshape = memref.reshape %alloc(%0) : (memref<1x1x1024xf32>, memref<4xi64>) -> memref<1x16x1x64xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1024xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 1024 {
          affine.store %cst_0, %alloc_1[%arg8, %arg9, %arg10] : memref<1x1x1024xf32>
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 1024 {
          affine.for %arg11 = 0 to 1024 {
            %1 = affine.load %arg3[%arg8, %arg9, %arg11] : memref<1x1x1024xf32, strided<[?, ?, ?], offset: ?>>
            %2 = affine.load %arg1[%arg8, %arg11, %arg10] : memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>
            %3 = affine.load %alloc_1[%arg8, %arg9, %arg10] : memref<1x1x1024xf32>
            %4 = arith.mulf %1, %2 : f32
            %5 = arith.addf %3, %4 : f32
            affine.store %5, %alloc_1[%arg8, %arg9, %arg10] : memref<1x1x1024xf32>
          }
        }
      }
    }
    %reshape_2 = memref.reshape %alloc_1(%0) : (memref<1x1x1024xf32>, memref<4xi64>) -> memref<1x16x1x64xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x1x1024xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 1024 {
          affine.store %cst_0, %alloc_3[%arg8, %arg9, %arg10] : memref<1x1x1024xf32>
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 1 {
        affine.for %arg10 = 0 to 1024 {
          affine.for %arg11 = 0 to 1024 {
            %1 = affine.load %arg3[%arg8, %arg9, %arg11] : memref<1x1x1024xf32, strided<[?, ?, ?], offset: ?>>
            %2 = affine.load %arg2[%arg8, %arg11, %arg10] : memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>
            %3 = affine.load %alloc_3[%arg8, %arg9, %arg10] : memref<1x1x1024xf32>
            %4 = arith.mulf %1, %2 : f32
            %5 = arith.addf %3, %4 : f32
            affine.store %5, %alloc_3[%arg8, %arg9, %arg10] : memref<1x1x1024xf32>
          }
        }
      }
    }

    
    
    %reshape_4 = memref.reshape %alloc_3(%0) : (memref<1x1x1024xf32>, memref<4xi64>) -> memref<1x16x1x64xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 64 {
            %1 = affine.apply #map(%arg10)[%arg6]
            %2 = affine.load %reshape_2[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x64xf32>
            affine.store %2, %arg4[%arg8, %arg9, %1, %arg11] : memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>
          }
        }
      }
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x16x512x64xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 512 {
          affine.for %arg11 = 0 to 64 {
            affine.store %cst_0, %alloc_5[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32>
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 512 {
          affine.for %arg11 = 0 to 64 {
            %1 = affine.load %arg4[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>
            affine.store %1, %alloc_5[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32>
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = %arg7 to 512 {
          affine.for %arg11 = 0 to 64 {
            affine.store %cst_0, %alloc_5[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32>
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 64 {
            %1 = affine.apply #map(%arg10)[%arg6]
            %2 = affine.load %reshape_4[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x64xf32>
            affine.store %2, %arg5[%arg8, %arg9, %1, %arg11] : memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>
          }
        }
      }
    }
    
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x16x512x64xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 512 {
          affine.for %arg11 = 0 to 64 {
            affine.store %cst_0, %alloc_6[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32>
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 512 {
          affine.for %arg11 = 0 to 64 {
            %1 = affine.load %arg5[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>
            affine.store %1, %alloc_6[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32>
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = %arg7 to 512 {
          affine.for %arg11 = 0 to 64 {
            affine.store %cst_0, %alloc_6[%arg8, %arg9, %arg10, %arg11] : memref<1x16x512x64xf32>
          }
        }
      }
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x16x64x512xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 64 {
          affine.for %arg11 = 0 to 512 {
            affine.store %cst_0, %alloc_7[%arg8, %arg9, %arg10, %arg11] : memref<1x16x64x512xf32>
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 64 {
          affine.for %arg11 = 0 to 512 {
            %1 = affine.load %alloc_5[%arg8, %arg9, %arg11, %arg10] : memref<1x16x512x64xf32>
            affine.store %1, %alloc_7[%arg8, %arg9, %arg10, %arg11] : memref<1x16x64x512xf32>
          }
        }
      }
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x16x1x512xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 512 {
            affine.store %cst_0, %alloc_8[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x512xf32>
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 64 {
            affine.for %arg12 = 0 to 512 {
              %1 = affine.load %reshape[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x64xf32>
              %2 = affine.load %alloc_7[%arg8, %arg9, %arg11, %arg12] : memref<1x16x64x512xf32>
              %3 = affine.load %alloc_8[%arg8, %arg9, %arg10, %arg12] : memref<1x16x1x512xf32>
              %4 = arith.mulf %1, %2 : f32
              %5 = arith.addf %3, %4 : f32
              affine.store %5, %alloc_8[%arg8, %arg9, %arg10, %arg12] : memref<1x16x1x512xf32>
            }
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = %arg7 to 512 {
            affine.store %cst, %alloc_8[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x512xf32>
          }
        }
      }
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x16x1x512xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 512 {
            %1 = affine.load %alloc_8[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x512xf32>
            %2 = math.exp %1 : f32
            affine.store %2, %alloc_9[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x512xf32>
          }
        }
      }
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x16x1xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 1 {
          affine.store %cst_0, %alloc_10[%arg8, %arg9, %arg10] : memref<1x16x1xf32>
        }
      }
    }
    
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 512 {
            %1 = affine.load %alloc_9[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x512xf32>
            %2 = affine.load %alloc_10[%arg8, %arg9, %arg10] : memref<1x16x1xf32>
            %3 = arith.addf %1, %2 : f32
            affine.store %3, %alloc_10[%arg8, %arg9, %arg10] : memref<1x16x1xf32>
          }
        }
      }
    }
    
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x16x1x512xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 512 {
            %1 = affine.load %alloc_10[%arg8, %arg9, %arg10] : memref<1x16x1xf32>
            affine.store %1, %alloc_11[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x512xf32>
          }
        }
      }
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x16x1x512xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 512 {
            %1 = affine.load %alloc_9[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x512xf32>
            %2 = affine.load %alloc_11[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x512xf32>
            %3 = arith.divf %1, %2 : f32
            affine.store %3, %alloc_12[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x512xf32>
          }
        }
      }
    }
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = %arg7 to 512 {
            affine.store %cst_0, %alloc_12[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x512xf32>
          }
        }
      }
    }
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x16x1x64xf32>
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 64 {
            affine.store %cst_0, %alloc_13[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x64xf32>
          }
        }
      }
    }
    
    %mi_cast = memref.cast %alloc_12: memref<1x16x1x512xf32> to memref<*xf32>
    func.call @printMemrefF32(%mi_cast) : (memref<*xf32>) -> ()
    affine.for %arg8 = 0 to 1 {
      affine.for %arg9 = 0 to 16 {
        affine.for %arg10 = 0 to 1 {
          affine.for %arg11 = 0 to 512 {
            affine.for %arg12 = 0 to 64 {
              %1 = affine.load %alloc_12[%arg8, %arg9, %arg10, %arg11] : memref<1x16x1x512xf32>
              %2 = affine.load %alloc_6[%arg8, %arg9, %arg11, %arg12] : memref<1x16x512x64xf32>
              %3 = affine.load %alloc_13[%arg8, %arg9, %arg10, %arg12] : memref<1x16x1x64xf32>
              %4 = arith.mulf %1, %2 : f32
              %5 = arith.addf %3, %4 : f32
              affine.store %5, %alloc_13[%arg8, %arg9, %arg10, %arg12] : memref<1x16x1x64xf32>
            }
          }
        }
      }
    }
    %mid_cast = memref.cast %alloc_13: memref<1x16x1x64xf32> to memref<*xf32>
    func.call @printMemrefF32(%mid_cast) : (memref<*xf32>) -> ()
    return %alloc_13 : memref<1x16x1x64xf32>
  }
  func.func @main() {
  // === Input Initialization ===
  // Wq, Wk, Wv : [1x1024x1024xf32]
  %Wq = memref.alloc() : memref<1x1024x1024xf32>
  %Wk = memref.alloc() : memref<1x1024x1024xf32>
  %Wv = memref.alloc() : memref<1x1024x1024xf32>
  affine.for %i = 0 to 1 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %cst = arith.constant 0.01 : f32
        affine.store %cst, %Wq[%i, %j, %k] : memref<1x1024x1024xf32>
        affine.store %cst, %Wk[%i, %j, %k] : memref<1x1024x1024xf32>
        affine.store %cst, %Wv[%i, %j, %k] : memref<1x1024x1024xf32>
      }
    }
  }

  // x : [1x256x1024xf32]
  %x = memref.alloc() : memref<1x256x1024xf32>
  affine.for %i = 0 to 1 {
    affine.for %j = 0 to 256 {
      affine.for %k = 0 to 1024 {
        %c1 = arith.constant 0.01 : f32
        affine.store %c1, %x[%i, %j, %k] : memref<1x256x1024xf32>
      }
    }
  }

  // K_cache, V_cache : [1x16x512x64xf32]
  %K_cache = memref.alloc() : memref<1x16x512x64xf32>
  %V_cache = memref.alloc() : memref<1x16x512x64xf32>
  affine.for %i = 0 to 1 {
    affine.for %j = 0 to 16 {
      affine.for %k = 0 to 512 {
        affine.for %l = 0 to 64 {
          %zero = arith.constant 0.0 : f32
          affine.store %zero, %K_cache[%i, %j, %k, %l] : memref<1x16x512x64xf32>
          affine.store %zero, %V_cache[%i, %j, %k, %l] : memref<1x16x512x64xf32>
        }
      }
    }
  }

  // === Iteration index ===
  %iter = arith.constant 0 : index
  %final_iter = arith.constant 256 : index

  // === Casts to match @qkt_computaton signature ===
  %Wq_cast = memref.cast %Wq : memref<1x1024x1024xf32> to memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>
  %Wk_cast = memref.cast %Wk : memref<1x1024x1024xf32> to memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>
  %Wv_cast = memref.cast %Wv : memref<1x1024x1024xf32> to memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>
  %x_cast  = memref.cast %x  : memref<1x256x1024xf32>   to memref<1x256x1024xf32, strided<[?, ?, ?], offset: ?>>
  %Kc_cast = memref.cast %K_cache : memref<1x16x512x64xf32> to memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>
  %Vc_cast = memref.cast %V_cache : memref<1x16x512x64xf32> to memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>

  %prompt_iter = arith.constant 0 : index
  %final_prompt_iter = arith.constant 256 : index
  // === Call qkt_computaton ===
  %result = func.call @qkt_computaton(
      %Wq_cast, %Wk_cast, %Wv_cast, %x_cast, %Kc_cast, %Vc_cast, %prompt_iter, %final_prompt_iter
    ) : (memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>,
         memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>,
         memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>,
         memref<1x256x1024xf32, strided<[?, ?, ?], offset: ?>>,
         memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>,
         memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>, index, index) -> memref<1x16x256x64xf32>
  
  // === Print Result ===

  %x_new = memref.alloc() : memref<1x1x1024xf32>
  affine.for %i = 0 to 1 {
    affine.for %j = 0 to 1 {
      affine.for %k = 0 to 1024 {
        %c1 = arith.constant 0.01 : f32
        affine.store %c1, %x_new[%i, %j, %k] : memref<1x1x1024xf32>
      }
    }
  }
   %x_newcast  = memref.cast %x_new  : memref<1x1x1024xf32>   to memref<1x1x1024xf32, strided<[?, ?, ?], offset: ?>>
   

  %token_iter = arith.constant 256 : index
  %final_token_iter = arith.constant 257 : index
  affine.for %t = 0 to 10 {
    %new_iter = arith.addi %token_iter, %t : index
    %new_final_iter = arith.addi %final_token_iter, %t : index
    %inner_result = func.call @token_computaton(
      %Wq_cast, %Wk_cast, %Wv_cast, %x_newcast, %Kc_cast, %Vc_cast, %new_iter, %new_final_iter
    ) : (memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>,
         memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>,
         memref<1x1024x1024xf32, strided<[?, ?, ?], offset: ?>>,
         memref<1x1x1024xf32, strided<[?, ?, ?], offset: ?>>,
         memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>,
         memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>>, index, index) -> memref<1x16x1x64xf32>
    // Here, you can add code to print or process inner_result as needed.
      //%res_cast = memref.cast %inner_result : memref<1x16x1x64xf32> to memref<*xf32>
      //func.call @printMemrefF32(%res_cast) : (memref<*xf32>) -> ()
      %res_cast = memref.cast %inner_result: memref<1x16x1x64xf32> to memref<*xf32>
    func.call @printMemrefF32(%res_cast) : (memref<*xf32>) -> ()
    }

    //%res_cast = memref.cast %Kc_cast : memref<1x16x512x64xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<*xf32>
    //  func.call @printMemrefF32(%res_cast) : (memref<*xf32>) -> ()

    
    
    

  return
  }
}

