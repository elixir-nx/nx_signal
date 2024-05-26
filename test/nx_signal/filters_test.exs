defmodule NxSignal.FiltersTest do
  use NxSignal.Case
  doctest NxSignal.Filters

  describe "median/2" do
    test "performs 1D median filter" do
      t = Nx.tensor([10, 9, 8, 7, 1, 4, 5, 3, 2, 6])
      opts = [kernel_shape: {3}]
      expected = Nx.tensor([9.0, 8.0, 7.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0])

      assert NxSignal.Filters.median(t, opts) == expected
    end

    test "performs 2D median filter" do
      t =
        Nx.tensor([
          [31, 11, 17, 13, 1],
          [1, 3, 19, 23, 29],
          [19, 5, 7, 37, 2]
        ])

      opts = [kernel_shape: {3, 3}]

      expected =
        Nx.tensor([
          [11.0, 13.0, 17.0, 17.0, 17.0],
          [11.0, 13.0, 17.0, 17.0, 17.0],
          [11.0, 13.0, 17.0, 17.0, 17.0]
        ])

      assert NxSignal.Filters.median(t, opts) == expected
    end

    test "performs n-dim median filter" do
      t =
        Nx.tensor([
          [
            [31, 11, 17, 13, 1],
            [1, 3, 19, 23, 29],
            [19, 5, 7, 37, 2]
          ],
          [
            [19, 5, 7, 37, 2],
            [1, 3, 19, 23, 29],
            [31, 11, 17, 13, 1]
          ],
          [
            [1, 3, 19, 23, 29],
            [31, 11, 17, 13, 1],
            [19, 5, 7, 37, 2]
          ]
        ])

      k1 = {3, 3, 1}
      k2 = {3, 3, 3}

      expected1 =
        Nx.tensor([
          [
            [19.0, 5.0, 17.0, 23.0, 2.0],
            [19.0, 5.0, 17.0, 23.0, 2.0],
            [19.0, 5.0, 17.0, 23.0, 2.0]
          ],
          [
            [19.0, 5.0, 17.0, 23.0, 2.0],
            [19.0, 5.0, 17.0, 23.0, 2.0],
            [19.0, 5.0, 17.0, 23.0, 2.0]
          ],
          [
            [19.0, 5.0, 17.0, 23.0, 2.0],
            [19.0, 5.0, 17.0, 23.0, 2.0],
            [19.0, 5.0, 17.0, 23.0, 2.0]
          ]
        ])

      expected2 =
        Nx.tensor([
          [
            [11.0, 13.0, 17.0, 17.0, 17.0],
            [11.0, 13.0, 17.0, 17.0, 17.0],
            [11.0, 13.0, 17.0, 17.0, 17.0]
          ],
          [
            [11.0, 13.0, 17.0, 17.0, 17.0],
            [11.0, 13.0, 17.0, 17.0, 17.0],
            [11.0, 13.0, 17.0, 17.0, 17.0]
          ],
          [
            [11.0, 13.0, 17.0, 17.0, 17.0],
            [11.0, 13.0, 17.0, 17.0, 17.0],
            [11.0, 13.0, 17.0, 17.0, 17.0]
          ]
        ])

      assert NxSignal.Filters.median(t, kernel_shape: k1) == expected1
      assert NxSignal.Filters.median(t, kernel_shape: k2) == expected2
    end

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
  end
end
