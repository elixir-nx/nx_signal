defmodule NxSignal.ConvolutionTest do
  use NxSignal.Case
  # doctest NxSignal.Filters
  import NxSignal.Helpers
  alias NxSignal.Convolution
  import NxSignal.Convolution, [:convolve, 3]

  describe "convolve/4" do
    # These tests were adapted from https://github.com/numpy/numpy/blob/v2.1.0/numpy/_core/tests/test_numeric.py#L3573
    test "numpy object" do
      d = Nx.tensor(List.duplicate(1.0, 100))
      k = Nx.tensor(List.duplicate(1.0, 3))
      c = NxSignal.Convolution.convolve(d, k)[2..-3//1]
      o = Nx.tensor(List.duplicate(3, 98))
      assert_all_close(c, o)
    end

    # These tests were adapted from https://github.com/scipy/scipy/blob/v1.14.1/scipy/signal/tests/test_signaltools.py
    test "basic" do
      a = Nx.tensor([3, 4, 5, 6, 5, 4])
      b = Nx.tensor([1, 2, 3])
      c = NxSignal.Convolution.convolve(a, b, mode: "full")
      assert c == Nx.as_type(Nx.tensor([3, 10, 22, 28, 32, 32, 23, 12]), {:f, 32})
    end

    test "same" do
      a = Nx.tensor([3, 4, 5])
      b = Nx.tensor([1, 2, 3, 4])
      c = NxSignal.Convolution.convolve(a, b, mode: "same")
      assert c == Nx.as_type(Nx.tensor([10, 22, 34]), {:f, 32})
    end

    test "same eq" do
      a = Nx.tensor([3, 4, 5])
      b = Nx.tensor([1, 2, 3])
      c = NxSignal.Convolution.convolve(a, b, mode: "same")
      assert c == Nx.as_type(Nx.tensor([10, 22, 22]), {:f, 32})
    end

    test "complex" do
      a = Nx.tensor([Complex.new(1, 1), Complex.new(2, 1), Complex.new(3, 1)])
      b = Nx.tensor([Complex.new(1, 1), Complex.new(2, 1)])
      c = NxSignal.Convolution.convolve(a, b)

      assert c ==
               Nx.tensor([
                 Complex.new(0, 2),
                 Complex.new(2, 6),
                 Complex.new(5, 8),
                 Complex.new(5, 5)
               ])
    end

    test "zero rank" do
      a = Nx.tensor(1289)
      b = Nx.tensor(4567)
      c = NxSignal.Convolution.convolve(a, b)
      assert c == Nx.as_type(Nx.multiply(a, b), {:f, 32})
    end

    test "complex simple" do
      a = Nx.tensor([Complex.new(1, 1)])
      b = Nx.tensor([Complex.new(3, 4)])
      c = NxSignal.Convolution.convolve(a, b)
      assert c == Nx.tensor([Complex.new(-1, 7)])
    end

    test "fft_nd" do
      a = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      c = NxSignal.Transforms.fft_nd(a, axes: [0, 1], lengths: [2, 3])

      z =
        Nx.tensor([[21, Complex.new(-3, 1.732), Complex.new(-3, -1.732)], [-9, 0, 0]])

      assert_all_close(
        c,
        z
      )
    end

    test "fft_nd with padding" do
      a = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      c = NxSignal.Transforms.fft_nd(a, axes: [0, 1], lengths: [3, 3])

      z =
        Nx.tensor([
          [2.1e1, Complex.new(-3, 1.732), Complex.new(-3, -1.732)],
          [Complex.new(-1.5, -12.99), Complex.new(-1.11e-16, 1.732), Complex.new(-1.5, 0.866)],
          [Complex.new(-1.5, 12.99), Complex.new(-1.5, -0.866), Complex.new(-1.11e-16, -1.732)]
        ])

      assert_all_close(
        c,
        z
      )
    end

    test "broadcastable" do
      a = Nx.iota({3, 3, 3})
      b = Nx.iota({1, 1, 3})

      x = NxSignal.Convolution.convolve(a, b, method: "direct")
      y = NxSignal.Convolution.convolve(a, b, method: "fft")

      expected =
        Nx.tensor([
          [[0, 0, 1, 4, 4], [0, 3, 10, 13, 10], [0, 6, 19, 22, 16]],
          [[0, 9, 28, 31, 22], [0, 12, 37, 40, 28], [0, 15, 46, 49, 34]],
          [[0, 18, 55, 58, 40], [0, 21, 64, 67, 46], [0, 24, 73, 76, 52]]
        ])

      assert_all_close(x, expected)
      assert_all_close(y, expected)

      b = Nx.reshape(b, {1, 3, 1})

      x = NxSignal.Convolution.convolve(a, b, method: "direct")
      y = NxSignal.Convolution.convolve(a, b, method: "fft")

      expected =
        Nx.tensor([
          [[0, 0, 0], [0, 1, 2], [3, 6, 9], [12, 15, 18], [12, 14, 16]],
          [[0, 0, 0], [9, 10, 11], [30, 33, 36], [39, 42, 45], [30, 32, 34]],
          [[0, 0, 0], [18, 19, 20], [57, 60, 63], [66, 69, 72], [48, 50, 52]]
        ])

      assert_all_close(x, expected)
      assert_all_close(y, expected)

      b = Nx.reshape(b, {3, 1, 1})

      x = NxSignal.Convolution.convolve(a, b, method: "direct")
      y = NxSignal.Convolution.convolve(a, b, method: "fft")

      expected =
        Nx.tensor([
          [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
          [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
          [[9, 12, 15], [18, 21, 24], [27, 30, 33]],
          [[36, 39, 42], [45, 48, 51], [54, 57, 60]],
          [[36, 38, 40], [42, 44, 46], [48, 50, 52]]
        ])

      assert_all_close(x, expected)
      assert_all_close(y, expected)
    end

    test "single element" do
      a = Nx.tensor([4967])
      b = Nx.tensor([3920])
      c = convolve(a, b)
      assert c == Nx.as_type(Nx.multiply(a, b), {:f, 32})
    end

    test "2d arrays" do
      a = Nx.tensor([[1, 2, 3], [3, 4, 5]])
      b = Nx.tensor([[2, 3, 4], [4, 5, 6]])
      c = convolve(a, b)

      d =
        Nx.tensor([[2, 7, 16, 17, 12], [10, 30, 62, 58, 38], [12, 31, 58, 49, 30]])
        |> Nx.as_type({:f, 32})

      assert c == d
    end

    test "input swapping" do
      small =
        0..(8 - 1)
        |> Enum.to_list()
        |> Nx.tensor()
        |> Nx.reshape({2, 2, 2})

      big =
        0..(27 - 1)
        |> Enum.to_list()
        |> Enum.map(&Complex.new(0, &1))
        |> Nx.tensor()
        |> Nx.reshape({3, 3, 3})

      big_add =
        0..(27 - 1)
        |> Enum.to_list()
        |> Enum.reverse()
        |> Nx.tensor()
        |> Nx.reshape({3, 3, 3})

      big = Nx.add(big, big_add)

      out_array =
        [
          [
            [Complex.new(0, 0), Complex.new(26, 0), Complex.new(25, 1), Complex.new(24, 2)],
            [Complex.new(52, 0), Complex.new(151, 5), Complex.new(145, 11), Complex.new(93, 11)],
            [Complex.new(46, 6), Complex.new(133, 23), Complex.new(127, 29), Complex.new(81, 23)],
            [Complex.new(40, 12), Complex.new(98, 32), Complex.new(93, 37), Complex.new(54, 24)]
          ],
          [
            [
              Complex.new(104, 0),
              Complex.new(247, 13),
              Complex.new(237, 23),
              Complex.new(135, 21)
            ],
            [
              Complex.new(282, 30),
              Complex.new(632, 96),
              Complex.new(604, 124),
              Complex.new(330, 86)
            ],
            [
              Complex.new(246, 66),
              Complex.new(548, 180),
              Complex.new(520, 208),
              Complex.new(282, 134)
            ],
            [
              Complex.new(142, 66),
              Complex.new(307, 161),
              Complex.new(289, 179),
              Complex.new(153, 107)
            ]
          ],
          [
            [
              Complex.new(68, 36),
              Complex.new(157, 103),
              Complex.new(147, 113),
              Complex.new(81, 75)
            ],
            [
              Complex.new(174, 138),
              Complex.new(380, 348),
              Complex.new(352, 376),
              Complex.new(186, 230)
            ],
            [
              Complex.new(138, 174),
              Complex.new(296, 432),
              Complex.new(268, 460),
              Complex.new(138, 278)
            ],
            [
              Complex.new(70, 138),
              Complex.new(145, 323),
              Complex.new(127, 341),
              Complex.new(63, 197)
            ]
          ],
          [
            [
              Complex.new(32, 72),
              Complex.new(68, 166),
              Complex.new(59, 175),
              Complex.new(30, 100)
            ],
            [
              Complex.new(68, 192),
              Complex.new(139, 433),
              Complex.new(117, 455),
              Complex.new(57, 255)
            ],
            [
              Complex.new(38, 222),
              Complex.new(73, 499),
              Complex.new(51, 521),
              Complex.new(21, 291)
            ],
            [
              Complex.new(12, 144),
              Complex.new(20, 318),
              Complex.new(7, 331),
              Complex.new(0, 182)
            ]
          ]
        ]
        |> Nx.tensor()

      assert convolve(small, big, mode: "full") == out_array
      assert convolve(big, small, mode: "full") == out_array
      assert convolve(small, big, mode: "same") == out_array[[1..2, 1..2, 1..2]]
      assert convolve(big, small, mode: "same") == out_array[[0..2, 0..2, 0..2]]
      assert convolve(small, big, mode: "valid") == out_array[[1..2, 1..2, 1..2]]
      assert convolve(big, small, mode: "valid") == out_array[[1..2, 1..2, 1..2]]
    end

    test "invalid params" do
      a = Nx.tensor([3, 4, 5])
      b = Nx.tensor([1, 2, 3])

      assert_raise(CaseClauseError, fn ->
        convolve(a, b, mode: "spam")
      end)

      assert_raise(FunctionClauseError, fn ->
        convolve(a, b, mode: "eggs", method: "fft")
      end)

      assert_raise(CaseClauseError, fn ->
        convolve(a, b, mode: "ham", method: "direct")
      end)

      assert_raise(CaseClauseError, fn ->
        convolve(a, b, mode: "full", method: "bacon")
      end)

      assert_raise(CaseClauseError, fn ->
        convolve(a, b, mode: "same", method: "bacon")
      end)
    end

    test "valid mode 2.1" do
      a = Nx.tensor([1, 2, 3, 6, 5, 3])
      b = Nx.tensor([2, 3, 4, 5, 3, 4, 2, 2, 1])
      expected = Nx.tensor([70, 78, 73, 65]) |> Nx.as_type({:f, 32})

      out = convolve(a, b, mode: "valid")
      assert out == expected

      out = convolve(b, a, mode: "valid")
      assert out == expected
    end

    test "valid mode 2.2" do
      a = Nx.tensor([Complex.new(1, 5), Complex.new(2, -1), Complex.new(3, 0)])
      b = Nx.tensor([Complex.new(2, -3), Complex.new(1, 0)])
      expected = Nx.tensor([Complex.new(2, -3), Complex.new(8, -10)])

      out = convolve(a, b, mode: "valid")
      assert out == expected

      out = convolve(b, a, mode: "valid")
      assert out == expected
    end
  end
end
