defmodule NxSignal.TransformsTests do
  use NxSignal.Case, async: true, validate_doc_metadata: false
  import NxSignal.Transforms

  describe "fftnd/2" do
    test "equivlanet to fft" do
      a = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      assert Nx.fft(a) == fft_nd(a)
    end

    test "all axes" do
      a = Nx.tensor([[1, 0], [0, 1]])

      out =
        Nx.tensor([
          [2, Complex.new(0, 0)],
          [Complex.new(0, 0), 2]
        ])

      assert fft_nd(a, axes: Nx.axes(a)) == out
    end
  end

  describe "ifftnd_/2" do
    test "equivlanet to ifft" do
      a = Nx.tensor([[1, 2, 3], [4, 5, 6]])
      assert Nx.ifft(a) == ifft_nd(a)
    end

    test "all axes" do
      a =
        Nx.tensor([
          [2, Complex.new(0, 0)],
          [Complex.new(0, 0), 2]
        ])

      out =
        Nx.tensor([[Complex.new(1, 0), 0], [0, 1]])

      assert ifft_nd(a, axes: Nx.axes(a)) == out
    end
  end
end
