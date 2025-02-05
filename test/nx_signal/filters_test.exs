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

  describe "wiener/2" do
    test "performs n-dim wiener filter with calculated noise" do
      im =
        Nx.tensor(
          [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0]
          ],
          type: :f64
        )

      kernel_size = {3, 3}

      expected =
        Nx.tensor(
          [
            [
              1.7777777777777777,
              3.0,
              3.6666666666666665,
              4.333333333333333,
              3.111111111111111
            ],
            [4.3366520642506305, 7.0, 8.0, 9.0, 7.58637597408283],
            [
              4.692197051420351,
              7.261706150595039,
              8.748939779474131,
              10.157992415073023,
              9.813815742524799
            ]
          ],
          type: :f64
        )

      assert NxSignal.Filters.wiener(im, kernel_size: kernel_size) == expected
      assert NxSignal.Filters.wiener(im, kernel_size: 3) == expected

      assert NxSignal.Filters.wiener(Nx.as_type(im, :f32), kernel_size: kernel_size) ==
               Nx.tensor([
                 [
                   1.7777777910232544,
                   3.0,
                   3.6666667461395264,
                   4.333333492279053,
                   3.1111111640930176
                 ],
                 [4.3366522789001465, 7.0, 8.0, 9.0, 7.586376190185547],
                 [
                   4.692196846008301,
                   7.261706352233887,
                   8.748939514160156,
                   10.157992362976074,
                   9.81381607055664
                 ]
               ])
    end

    test "performs n-dim wiener filter with parameterized noise" do
      im =
        Nx.tensor(
          [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0]
          ],
          type: :f64
        )

      kernel_size = {3, 3}

      assert NxSignal.Filters.wiener(im, kernel_size: kernel_size, noise: 10) ==
               Nx.tensor(
                 [
                   [
                     1.7777777777777777,
                     3.0,
                     3.5882352941176467,
                     4.238095238095238,
                     3.7397034596375622
                   ],
                   [5.193548387096774, 7.0, 8.0, 9.0, 8.829787234042554],
                   [
                     7.941747572815534,
                     9.702702702702702,
                     10.938931297709924,
                     12.137254901960784,
                     12.485549132947977
                   ]
                 ],
                 type: :f64
               )

      assert NxSignal.Filters.wiener(Nx.as_type(im, :f32), kernel_size: kernel_size, noise: 10) ==
               Nx.tensor([
                 [
                   1.7777777910232544,
                   3.0,
                   3.588235378265381,
                   4.238095283508301,
                   3.739703416824341
                 ],
                 [5.193548202514648, 7.0, 8.0, 9.0, 8.829787254333496],
                 [
                   7.941747665405273,
                   9.702702522277832,
                   10.938931465148926,
                   12.13725471496582,
                   12.485548973083496
                 ]
               ])

      assert NxSignal.Filters.wiener(im, kernel_size: kernel_size, noise: 0) ==
               Nx.tensor(
                 [
                   [1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0],
                   [11.0, 12.0, 13.0, 14.0, 15.0]
                 ],
                 type: :f64
               )
    end
  end
end
