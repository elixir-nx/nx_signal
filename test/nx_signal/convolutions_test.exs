defmodule NxSignal.ConvolutionTest do
  use NxSignal.Case
  # doctest NxSignal.Filters

  describe "convolve/4" do
    # These tests were adapted from https://github.com/scipy/scipy/blob/v1.14.1/scipy/signal/tests/test_signaltools.py
    test "basic" do
      a = Nx.tensor([3, 4, 5, 6, 5, 4])
      b = Nx.tensor([1, 2, 3])
      c = NxSignal.Convolution.convolve(a, b)
      assert c == Nx.tensor([3, 10, 22, 28, 32, 32, 23, 12])
    end
  end
end
