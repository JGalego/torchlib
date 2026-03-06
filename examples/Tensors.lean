import TorchLib.Core

/-!
# Example: Tensors and Shapes

Demonstrates the core `Tensor` and `Shape` API from `TorchLib.Core`.
-/

open TorchLib

-- ---------------------------------------------------------------------------
-- Construction
-- ---------------------------------------------------------------------------

-- A 3×4 zero-filled tensor
def zeros34 : Tensor Float := Tensor.zeros [3, 4]

-- A 2×3 tensor filled with 0.5
def half23 : Tensor Float := Tensor.full [2, 3] 0.5

-- A 1-D ones vector of length 5
def ones5 : Tensor Float := Tensor.ones [5]

#eval zeros34.shape   -- [3, 4]
#eval zeros34.numel   -- 12
#eval zeros34.rank    -- 2

#eval half23.data     -- 6 entries, all 0.5

-- ---------------------------------------------------------------------------
-- Arithmetic
-- ---------------------------------------------------------------------------

def x : Tensor Float := Tensor.full [4] 2.0
def y : Tensor Float := Tensor.full [4] 3.0

#eval (x + y).data    -- #[5.0, 5.0, 5.0, 5.0]
#eval (x * y).data    -- #[6.0, 6.0, 6.0, 6.0]
#eval (y - x).data    -- #[1.0, 1.0, 1.0, 1.0]

-- Scalar multiply
#eval (Tensor.smul 4.0 x).data   -- #[8.0, 8.0, 8.0, 8.0]

-- ---------------------------------------------------------------------------
-- Element-wise map / zipWith
-- ---------------------------------------------------------------------------

-- Apply ReLU manually
def relu5 : Tensor Float :=
  Tensor.full [5] (-1.0) |>.map (fun v => if v > 0.0 then v else 0.0)

#eval relu5.data   -- #[0.0, 0.0, 0.0, 0.0, 0.0]

-- Apply the `Scalar.relu` from the typeclass
def reluPos : Tensor Float := x.map Scalar.relu

#eval reluPos.data   -- #[2.0, 2.0, 2.0, 2.0]

-- ---------------------------------------------------------------------------
-- Reshape and flatten
-- ---------------------------------------------------------------------------

def mat24 : Tensor Float := Tensor.ones [2, 4]

-- Reshape to [4, 2]
#eval (mat24.reshape [4, 2]).map (·.shape)   -- some [4, 2]

-- Flatten to [8]
#eval mat24.flatten.shape   -- [8]

-- ---------------------------------------------------------------------------
-- 2-D transpose
-- ---------------------------------------------------------------------------

-- Build a [2, 3] tensor with distinct values
def mat23 : Tensor Float :=
  { shape := [2, 3]
    data  := #[1, 2, 3, 4, 5, 6] }

-- After transpose: shape [3, 2], data in column-major order
#eval mat23.transpose.shape   -- [3, 2]
#eval mat23.transpose.data    -- #[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

-- ---------------------------------------------------------------------------
-- Matrix multiplication  [m,k] × [k,n] → [m,n]
-- ---------------------------------------------------------------------------

-- Identity-like  [2,2] × [2,2]
def I2 : Tensor Float := { shape := [2, 2], data := #[1, 0, 0, 1] }
def A2 : Tensor Float := { shape := [2, 2], data := #[1, 2, 3, 4] }

#eval (Tensor.matmul A2 I2).data   -- #[1.0, 2.0, 3.0, 4.0]
