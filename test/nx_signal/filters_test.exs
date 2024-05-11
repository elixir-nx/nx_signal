defmodule NxSignal.FiltersTest do
  use NxSignal.Case
  doctest NxSignal.Filters

  describe "median/2" do
    test "raises if kernel_shape is not compatible" do
      t1 = Nx.iota({10})
      opts1 = [kernel_shape: {5, 5}]

      assert_raise(
        ArgumentError,
        "kernel shape must be of the same rank as the tensor",
        fn -> NxSignal.Filters.median(t1, opts1) end
      )

      t2 = Nx.iota({5, 5})
      opts2 = [kernel_shape: {5, 5, 5}]

      assert_raise(
        ArgumentError,
        "kernel shape must be of the same rank as the tensor",
        fn -> NxSignal.Filters.median(t2, opts2) end
      )
    end

    test "raises if tensor rank is not 1 or 2" do
      t1 = Nx.tensor(1)
      opts1 = [kernel_shape: {1}]

      assert_raise(
        ArgumentError,
        "tensor must be of rank 1 or 2",
        fn -> NxSignal.Filters.median(t1, opts1) end
      )

      t2 = Nx.iota({5, 5, 5})
      opts2 = [kernel_shape: {3, 3, 3}]

      assert_raise(
        ArgumentError,
        "tensor must be of rank 1 or 2",
        fn -> NxSignal.Filters.median(t2, opts2) end
      )
    end
  end
end
