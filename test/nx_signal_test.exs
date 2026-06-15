defmodule NxSignalTest do
  use NxSignal.Case
  doctest NxSignal

  describe "czt/2" do
    test "default params equal DFT" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      result = NxSignal.czt(x)
      expected = Nx.fft(Nx.as_type(x, {:c, 64}))

      assert_all_close(result, expected, atol: 1.0e-5)
    end

    test "unit impulse returns all ones" do
      x = Nx.tensor([1.0, 0.0, 0.0, 0.0])
      result = NxSignal.czt(x, m: 6)
      ones = Nx.broadcast(Nx.complex(1.0, 0.0), {6})

      assert_all_close(result, ones, atol: 1.0e-5)
    end

    test "M > N is equivalent to zero-padded DFT" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      result = NxSignal.czt(x, m: 8)
      expected = Nx.fft(Nx.as_type(x, {:c, 64}), length: 8)

      assert_all_close(result, expected, atol: 1.0e-5)
    end

    test "custom a evaluates z-transform off the unit circle" do
      # x = [1, 1], so X(z) = 1 + z^{-1}. At z = 2: X(2) = 1 + 0.5 = 1.5
      x = Nx.tensor([1.0, 1.0])
      a = Nx.complex(2.0, 0.0)
      w = Nx.complex(-1.0, 0.0)
      result = NxSignal.czt(x, m: 1, a: a, w: w)

      assert_all_close(Nx.real(result[0]), Nx.tensor(1.5, type: {:f, 64}), atol: 1.0e-5)
    end

    test "returns complex output" do
      result = NxSignal.czt(Nx.tensor([1.0, 2.0, 3.0]))

      assert Nx.Type.complex?(Nx.type(result))
    end
  end

  describe "zoom_fft/4" do
    test "full range [0, 1] equals DFT" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0])
      result = NxSignal.zoom_fft(x, 0.0, 1.0)
      expected = Nx.fft(Nx.as_type(x, {:c, 64}))

      assert_all_close(result, expected, atol: 1.0e-5)
    end

    test "resolves two closely-spaced tones" do
      # Two tones at normalised frequencies 0.49 and 0.51.
      # A 16-point DFT cannot separate them; zoom_fft over [0.45, 0.55] can.
      n = 128
      t = Nx.iota({n}, type: :f64)
      pi2 = 2 * :math.pi()

      x =
        Nx.add(
          Nx.cos(Nx.multiply(t, pi2 * 0.49)),
          Nx.cos(Nx.multiply(t, pi2 * 0.51))
        )

      result = NxSignal.zoom_fft(x, 0.45, 0.55, m: 64)
      magnitudes = Nx.abs(result)

      max_val = Nx.reduce_max(magnitudes) |> Nx.to_number()
      median_val = Nx.median(magnitudes) |> Nx.to_number()

      assert max_val > 5 * median_val
    end

    test "returns m output points" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
      result = NxSignal.zoom_fft(x, 0.1, 0.4, m: 16)

      assert Nx.shape(result) == {16}
    end
  end
end
