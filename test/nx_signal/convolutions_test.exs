defmodule NxSignal.ConvolutionTest do
  use NxSignal.Case
  # doctest NxSignal.Filters
  import NxSignal.Helpers

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
  end
end
