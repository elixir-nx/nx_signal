defmodule NxSignal.ConvolutionTest do
  use NxSignal.Case
  # doctest NxSignal.Filters

  describe "convolve/4" do
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
  end
end
