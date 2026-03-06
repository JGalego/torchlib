import TorchLib.Core

/-!
# Example: Tensors and Shapes

Demonstrates the core `Tensor` and `Shape` API from `TorchLib.Core`.
-/

open TorchLib

private def say [Repr α] (label : String) (v : α) : IO Unit :=
  IO.println s!"{label}: {reprStr v}"

-- ---------------------------------------------------------------------------
-- Construction
-- ---------------------------------------------------------------------------

-- A 3×4 zero-filled tensor
def zeros34 : Tensor Float := Tensor.zeros [3, 4]

-- A 2×3 tensor filled with 0.5
def half23 : Tensor Float := Tensor.full [2, 3] 0.5

-- A 1-D ones vector of length 5
def ones5 : Tensor Float := Tensor.ones [5]

#eval say "zeros34.shape" zeros34.shape
#eval say "zeros34.numel" zeros34.numel
#eval say "zeros34.rank"  zeros34.rank

#eval say "half23.data"   half23.data

-- ---------------------------------------------------------------------------
-- Arithmetic
-- ---------------------------------------------------------------------------

def x : Tensor Float := Tensor.full [4] 2.0
def y : Tensor Float := Tensor.full [4] 3.0

#eval say "x + y"   (x + y).data
#eval say "x * y"   (x * y).data
#eval say "y - x"   (y - x).data

-- Scalar multiply
#eval say "4.0 * x" (Tensor.smul 4.0 x).data

-- ---------------------------------------------------------------------------
-- Element-wise map / zipWith
-- ---------------------------------------------------------------------------

-- Apply ReLU manually
def relu5 : Tensor Float :=
  Tensor.full [5] (-1.0) |>.map (fun v => if v > 0.0 then v else 0.0)

#eval say "relu5.data"   relu5.data

-- Apply the `Scalar.relu` from the typeclass
def reluPos : Tensor Float := x.map Scalar.relu

#eval say "reluPos.data" reluPos.data

-- ---------------------------------------------------------------------------
-- Reshape and flatten
-- ---------------------------------------------------------------------------

def mat24 : Tensor Float := Tensor.ones [2, 4]

-- Reshape to [4, 2]
#eval say "mat24.reshape [4,2]"  ((mat24.reshape [4, 2]).map (·.shape))

-- Flatten to [8]
#eval say "mat24.flatten.shape" mat24.flatten.shape

-- ---------------------------------------------------------------------------
-- 2-D transpose
-- ---------------------------------------------------------------------------

-- Build a [2, 3] tensor with distinct values
def mat23 : Tensor Float :=
  { shape := [2, 3]
    data  := #[1, 2, 3, 4, 5, 6] }

-- After transpose: shape [3, 2], data in column-major order
#eval say "mat23.transpose.shape" mat23.transpose.shape
#eval say "mat23.transpose.data"  mat23.transpose.data

-- ---------------------------------------------------------------------------
-- Matrix multiplication  [m,k] × [k,n] → [m,n]
-- ---------------------------------------------------------------------------

-- Identity-like  [2,2] × [2,2]
def I2 : Tensor Float := { shape := [2, 2], data := #[1, 0, 0, 1] }
def A2 : Tensor Float := { shape := [2, 2], data := #[1, 2, 3, 4] }

#eval say "A2 @ I2" (Tensor.matmul A2 I2).data
