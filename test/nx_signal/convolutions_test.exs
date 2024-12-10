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

    test "broadcastable" do
      a = 0..26 |> Enum.to_list() |> Nx.tensor() |> Nx.reshape({3, 3, 3})

      b = 0..2 |> Enum.to_list() |> Nx.tensor()

      for i <- 0..2 do
        b_shape = [1, 1, 1]
        b_shape = List.replace_at(b_shape, i, 3)
        b_shape = List.to_tuple(b_shape)

        x =
          NxSignal.Convolution.convolve(a, Nx.reshape(b, b_shape), method: "direct")

        y =
          NxSignal.Convolution.convolve(a, Nx.reshape(b, b_shape), method: "fft")

        assert_all_close(x, y)
      end
    end
  end
end
